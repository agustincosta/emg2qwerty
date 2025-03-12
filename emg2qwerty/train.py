# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import set_float32_matmul_precision

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    set_float32_matmul_precision("high")
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        users = [session["user"] for session in dataset]
        if config.reduced:
            return [
                Path(config.dataset.root).joinpath(f"{user}_processed").joinpath(f"{session}.hdf5")
                for session, user in zip(sessions, users)
            ]
        else:
            return [Path(config.dataset.root).joinpath(f"{session}.hdf5") for session in sessions]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        checkpoint = torch.load(
            config.checkpoint, map_location=lambda storage, loc: storage, weights_only=False
        )
        module.load_state_dict(checkpoint["state_dict"])
        # module = module.load_from_checkpoint(
        #     config.checkpoint,
        #     optimizer=config.optimizer,
        #     lr_scheduler=config.lr_scheduler,
        #     decoder=config.decoder,
        # )

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks: list[pl.Callback] = []

    # Extract model name for checkpoint naming
    model_name = "model-multi-scale-autoencoder_24-tiny"
    log.info(f"Using model: {model_name}")

    # Process callbacks and customize ModelCheckpoint if present
    for cfg in callback_configs:
        if cfg._target_ == "pytorch_lightning.callbacks.ModelCheckpoint":
            # Customize the ModelCheckpoint callback
            checkpoint_callback = instantiate(
                cfg,
                filename=f"{model_name}-epoch{{epoch:02d}}-val_metric{{val_metric:.4f}}",
                monitor=config.monitor_metric,
                dirpath=f"{Path.cwd()}/checkpoints/{model_name}",
            )
            callbacks.append(checkpoint_callback)
        else:
            callbacks.append(instantiate(cfg))

    # Initialize trainer
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(save_dir=f"{Path.cwd()}/logs/", name=model_name),
            CSVLogger(save_dir=f"{Path.cwd()}/logs/", name=model_name),
        ],
    )

    if config.train:
        # Check if a past checkpoint exists to resume training from
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

        # Train
        trainer.fit(module, datamodule, ckpt_path=resume_from_checkpoint)

        # Load best checkpoint
        checkpoint = torch.load(
            trainer.checkpoint_callback.best_model_path,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        module.load_state_dict(checkpoint["state_dict"])
        # module = module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Validate and test on the best checkpoint (if training), or on the
    # loaded `config.checkpoint` (otherwise)
    val_metrics = trainer.validate(module, datamodule)
    test_metrics = trainer.test(module, datamodule)

    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path,
    }
    pprint.pprint(results, sort_dicts=False)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
