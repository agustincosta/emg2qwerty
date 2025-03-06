# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch
from dotenv import load_dotenv
from s3torchconnector import S3Checkpoint, S3ClientConfig
from torch import nn

from emg2qwerty.charset import CharacterSet, charset
from emg2qwerty.transforms import ToTensor, Transform

load_dotenv(override=True)


@dataclass
class EMGSessionData:
    """A read-only interface to a single emg2qwerty session file stored in
    HDF5 format.

    A session here refers to a span of a few minutes during which two-handed
    EMG signals were recorded while a user typed out a series of prompted
    words/sentences. This class encapsulates the EMG timeseries, ground-truth,
    and additional metadata corresponding to a single session.

    ``self.timeseries`` is a `h5py.Dataset` instance with a compound datatype
    as in a numpy structured array containing three fields - EMG data from the
    left and right wrists, and their corresponding timestamps.
    The sampling rate of EMG is 2kHz, each EMG device has 16 electrode
    channels, and the signal has been high-pass filtered. Therefore, the fields
    corresponding to left and right EMG are 2D arrays of shape ``(T, 16)`` each
    and ``timestamps`` is a 1D array of length ``T``.

    ``self.metadata`` contains two kinds of ground-truth:
      1. A sequence of ``prompts`` displayed to the user (where each prompt
         is a handful of words) along with their start and end timestamps.
         This offers less accurate ground-truth as there is no guarantee that
         the user actually typed out the prompted words accurately without typos.
         It also lacks time alignment of each key-press within the prompt window.
      2. A sequence of ``keystrokes`` indicating the key-presses on a keyboard
         as recorded by a keylogger, along with the timestamps corresponding to
         individual key-presses and the key-releases. This offers high-quality
         and accurate ground-truth as well as temporal alignment of EMG window
         with each key character.

    NOTE: Only the metadata and ground-truth are loaded into memory while the
    EMG data is accesssed directly from disk. When wrapping this interface
    within a PyTorch Dataset, use multiple dataloading workers to mask the
    disk seek and read latencies."""

    HDF5_GROUP: ClassVar[str] = "emg2qwerty"
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG_LEFT: ClassVar[str] = "emg_left"
    EMG_RIGHT: ClassVar[str] = "emg_right"
    TIMESTAMPS: ClassVar[str] = "time"
    SESSION_NAME: ClassVar[str] = "session_name"
    USER: ClassVar[str] = "user"
    CONDITION: ClassVar[str] = "condition"
    DURATION_MINS: ClassVar[str] = "duration_mins"
    KEYSTROKES: ClassVar[str] = "keystrokes"
    PROMPTS: ClassVar[str] = "prompts"

    hdf5_path: Path

    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        emg2qwerty_group: h5py.Group = self._file[self.HDF5_GROUP]

        # ``timeseries`` is a compound HDF5 Dataset with aligned left and right EMG
        # along with corresponding timestamps. Avoid loading the entire timeseries
        # into memory here - users should instead rely on PyTorch DataLoader workers
        # to mask the disk seek/read latency.
        self.timeseries: h5py.Dataset = emg2qwerty_group[self.TIMESERIES]
        assert self.timeseries.dtype.fields is not None
        assert self.EMG_LEFT in self.timeseries.dtype.fields
        assert self.EMG_RIGHT in self.timeseries.dtype.fields
        assert self.TIMESTAMPS in self.timeseries.dtype.fields

        # Load the metadata entirely into memory as it's rather small.
        self.metadata: dict[str, Any] = {}
        for key, val in emg2qwerty_group.attrs.items():
            if key in {self.KEYSTROKES, self.PROMPTS}:
                self.metadata[key] = json.loads(val)
            else:
                self.metadata[key] = val

    def __enter__(self) -> EMGSessionData:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()

    def __len__(self) -> int:
        return len(self.timeseries)

    def __getitem__(self, key: slice | str) -> np.ndarray:
        """Return a slice of the timeseries."""
        return self.timeseries[key]  # type: ignore

    def slice(self, start_t: float = -np.inf, end_t: float = np.inf) -> np.ndarray:
        """Load and return a contiguous slice of the timeseries windowed by the
        provided start and end timestamps.

        Args:
            start_t (float): The start time of the window to grab
                (in absolute unix time). Defaults to selecting from the
                beginning of the session. (default: ``-np.inf``).
            end_t (float): The end time of the window to grab
                (in absolute unix time). Defaults to selecting until the
                end of the session. (default: ``np.inf``)
        """
        start_idx, end_idx = self.timestamps.searchsorted([start_t, end_t])
        return self[start_idx:end_idx]

    def ground_truth(self, start_t: float = -np.inf, end_t: float = np.inf) -> LabelData:
        if self.condition == "on_keyboard":
            return LabelData.from_keystrokes(self.keystrokes, start_t=start_t, end_t=end_t)
        else:
            return LabelData.from_prompts(self.prompts, start_t=start_t, end_t=end_t)

    @property
    def fields(self) -> list[str]:
        """The names of the fields in the ``timeseries`` compound HDF5 Dataset."""
        return list(self.timeseries.dtype.fields.keys())

    @property
    def timestamps(self) -> np.ndarray:
        """EMG timestamps.

        NOTE: This reads the entire sequence of timesetamps from the underlying
        HDF5 file and therefore incurs disk latency. Avoid this in the critical
        path."""
        timestamps = self.timeseries[self.TIMESTAMPS]
        assert (np.diff(timestamps) >= 0).all(), "Timestamps are not monotonic"
        return timestamps  # type: ignore

    @property
    def session_name(self) -> str:
        """Unique name of the session."""
        return self.metadata[self.SESSION_NAME]  # type: ignore

    @property
    def user(self) -> str:
        """Unique ID of the user this session corresponds to."""
        return self.metadata[self.USER]  # type: ignore

    @property
    def condition(self) -> str:
        return self.metadata[self.CONDITION]  # type: ignore

    @property
    def duration_mins(self) -> float:
        """The duration of the EMG session in minutes."""
        return self.metadata[self.DURATION_MINS]  # type: ignore

    @property
    def keystrokes(self) -> list[dict[str, Any]]:
        """Sequence of keys recorded by the keylogger during the
        data-collection session along with the press and release timestamps
        for each key."""
        return self.metadata[self.KEYSTROKES]  # type: ignore

    @property
    def prompts(self) -> list[dict[str, Any]]:
        """Sequence of sentences prompted to the user during the
        data-collection session along with the start and end timestamps
        for each prompt."""
        return self.metadata[self.PROMPTS]  # type: ignore

    def __str__(self) -> str:
        """Human-readable string representation for display."""
        return (
            f"{self.__class__.__name__} {self.session_name}: "
            f"({len(self)} EMG samples, "
            f"{len(self.keystrokes)} keystrokes, "
            f"{self.duration_mins:.2f} mins)"
        )


@dataclass
class LabelData:
    """Canonical representation for text/label data together with optional
    character-level timestamps. Supports standarization from keylogger keystrokes,
    prompts, and pynput key representations.

    NOTE: Avoid calling ``LabelData`` constructor directly and instead
    use the provided factory classmethods as much as possible."""

    text: str
    _timestamps: InitVar[Sequence[float] | None] = None
    _charset: CharacterSet = field(default_factory=charset)

    def __post_init__(self, _timestamps: Sequence[float] | None) -> None:
        self.timestamps: np.ndarray | None = None
        if _timestamps is not None:
            self.timestamps = np.array(_timestamps, dtype=float)
            assert self.timestamps.ndim == 1
            assert len(self.timestamps) == len(self.text)
            assert (np.diff(self.timestamps) >= 0).all(), "Timestamps are not monotonic"

    @classmethod
    def from_keystrokes(
        cls,
        keystrokes: Sequence[Mapping[str, Any]],
        start_t: float = -np.inf,
        end_t: float = np.inf,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a sequence of
        keystrokes between the provided start and end timestamps, after
        normalizing and cleaning up as per ``CharacterSet``. The returned
        object also includes the key-press timestamps corresponding to each
        character in ``self.text``.

        Args:
            keystrokes (list): Each keystroke entry in the list should be a
                dict in the format of ``EMGSessionData.keystrokes``.
            start_t (float): The start timestamp of the window in absolute
                unix time. (default: ``-np.inf``)
            end_t (float): The end timestamp of the window in absolute
                unix time. (default: ``np.inf``)
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        label_data = cls(text="", _timestamps=[], _charset=_charset)
        for key in keystrokes:
            if key["start"] > end_t:
                break
            if key["start"] >= start_t:
                label_data += cls.from_key(key, _charset=_charset)
        return label_data

    @classmethod
    def from_key(
        cls,
        key: str | Mapping[str, Any],
        timestamp: float | None = None,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a single keystroke,
        after normalizing and cleaning up as per ``CharacterSet``.

        Args:
            key (str or dict): A single pynput.Key string or a keystroke
                dict in the format of ``EMGSessionData.keystrokes``.
            timestamp (float): Optional timestamp corresponding to the key.
                If not provided and if ``key`` is a dict, this will be set to the
                key-press time available in the dict. (default: ``None``)
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        if isinstance(key, str):
            _key = key
        else:
            _key = key["key"]
            timestamp = key["start"] if timestamp is None else timestamp

        _key = _charset._normalize_keys([_key])[0]
        if _key not in _charset:  # Out of vocabulary
            return cls(text="", _timestamps=[], _charset=_charset)

        text = _charset.keys_to_str([_key])
        timestamps = [timestamp] if timestamp is not None else None
        return cls(text, timestamps, _charset=_charset)

    @classmethod
    def from_prompts(
        cls,
        prompts: Sequence[Mapping[str, Any]],
        enforce_newline: bool = True,
        start_t: float = -np.inf,
        end_t: float = np.inf,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a sequence of prompts
        between the provided start and end timestamps, after normalizing and
        cleaning up as per ``CharacterSet``. The returned object does not
        include character-level timestamps.

        Assumes the input prompt sequence is sorted by time.

        Args:
            prompts (list): Each prompt entry in the list should be a dict in
                the format of ``EMGSessionData.prompts``.
            enforce_newline (bool): If set, end each prompt with a newline
                if not present already. (default: ``True``)
            start_t (float): The start timestamp of the window in absolute
                unix time. (default: ``-np.inf``)
            end_t (float): The end timestamp of the window in absolute
                unix time. (default: ``np.inf``)
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        label_data = cls(text="", _charset=_charset)
        for prompt in prompts:
            if prompt["start"] > end_t:
                break
            if prompt["start"] >= start_t:
                label_data += cls.from_prompt(
                    prompt,
                    enforce_newline=enforce_newline,
                    _charset=_charset,
                )
        return label_data

    @classmethod
    def from_prompt(
        cls,
        prompt: str | Mapping[str, Any],
        enforce_newline: bool = True,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a single prompt, after
        normalizing and cleaning up as per ``CharacterSet``. The returned
        object does not include character-level timestamps.

        Args:
            prompt (str or dict): A single prompt, either as raw text or a
                dict in the format of ``EMGSessionData.prompts``.
            enforce_newline (bool): If set, end the prompt with a newline
                if not present already. (default: ``True``)
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        if isinstance(prompt, str):
            text = prompt
        else:
            payload = prompt["payload"]
            text = payload["text"] if payload is not None else None

        # Do not add terminal newline if there was no prompt payload
        if text is None:
            return cls(text="", _charset=charset())

        text = _charset.clean_str(text)
        if enforce_newline and (len(text) == 0 or text[-1] != "⏎"):
            text += "⏎"
        return cls(text, _charset=_charset)

    @classmethod
    def from_str(
        cls,
        text: str,
        timestamps: Sequence[float] | None = None,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from a raw string, after
        normalizing and cleaning up as per ``CharacterSet``.

        Args:
            text (str): Raw text string to normalize and wrap into ``LabelData``.
            timestamps (list): Optional list of character-level timestamps of the
                same length as ``text``.
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        text = _charset.clean_str(text)
        return cls(text, timestamps, _charset=_charset)

    @classmethod
    def from_labels(
        cls,
        labels: Sequence[int],
        timestamps: Sequence[float] | None = None,
        _charset: CharacterSet | None = None,
    ) -> LabelData:
        """Create a new instance of ``LabelData`` from integer labels
        and optionally together with its corresponding timestamps.

        Args:
            labels (list): Sequene of integer labels belonging to CharacterSet.
            timestamps (list): Optional list of timestamps of the
                same length as ``labels``.
            _charset (CharacterSet): Optional ``CharacterSet`` override.
        """
        _charset = _charset or charset()

        text = _charset.labels_to_str(labels)
        return cls(text, timestamps, _charset=_charset)

    @property
    def labels(self) -> np.ndarray:
        """Integer labels corresponding to the label string."""
        labels = self._charset.str_to_labels(self.text)
        return np.asarray(labels, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.text)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LabelData):
            return NotImplemented
        return self.text == other.text

    def __add__(self, other: LabelData) -> LabelData:
        text = self.text + other.text
        if self.timestamps is not None and other.timestamps is not None:
            timestamps = np.append(self.timestamps, other.timestamps)
        else:
            timestamps = None

        return LabelData(text, timestamps, _charset=self._charset)

    def __str__(self) -> str:
        """Human-readable string representation for display."""
        return self.text.replace("⏎", "\n")


@dataclass
class WindowedEMGDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` corresponding to an instance of `EMGSessionData`
    that iterates over EMG windows of configurable length and stride.

    Args:
        hdf5_path (str): Path to the session file in hdf5 format.
        window_length (int): Size of each window. Specify None for no windowing
            in which case this will be a dataset of length 1 containing the
            entire session. (default: ``None``)
        stride (int): Stride between consecutive windows. Specify None to set
            this to window_length, in which case there will be no overlap
            between consecutive windows. (default: ``window_length``)
        padding (tuple[int, int]): Left and right contextual padding for
            windows in terms of number of raw EMG samples.
        jitter (bool): If True, randomly jitter the offset of each window.
            Use this for training time variability. (default: ``False``)
        transform (Callable): A composed sequence of transforms that takes
            a window/slice of `EMGSessionData` in the form of a numpy
            structured array and returns a `torch.Tensor` instance.
            (default: ``emg2qwerty.transforms.ToTensor()``)
    """

    hdf5_path: Path
    window_length: InitVar[int | None] = None
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False
    transform: Transform[np.ndarray, torch.Tensor] = field(default_factory=ToTensor)

    def __post_init__(
        self,
        window_length: int | None,
        stride: int | None,
        padding: tuple[int, int],
    ) -> None:
        with EMGSessionData(self.hdf5_path) as session:
            assert (
                session.condition == "on_keyboard"
            ), f"Unsupported condition {self.session.condition}"
            self.session_length = len(session)

        self.window_length = window_length if window_length is not None else self.session_length
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0

    def __len__(self) -> int:
        return int(max(self.session_length - self.window_length, 0) // self.stride + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy init `EMGSessionData` per dataloading worker
        # since `h5py.File` objects can't be picked.
        if not hasattr(self, "session"):
            self.session = EMGSessionData(self.hdf5_path)

        offset = idx * self.stride

        # Randomly jitter the window offset.
        leftover = len(self.session) - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Expand window to include contextual padding and fetch.
        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        window = self.session[window_start:window_end]

        # Extract EMG tensor corresponding to the window.
        emg = self.transform(window)
        assert torch.is_tensor(emg)

        # Extract labels corresponding to the original (un-padded) window.
        timestamps = window[EMGSessionData.TIMESTAMPS]
        start_t = timestamps[offset - window_start]
        end_t = timestamps[(offset + self.window_length - 1) - window_start]
        label_data = self.session.ground_truth(start_t, end_t)
        labels = torch.as_tensor(label_data.labels)

        return emg, labels

    @staticmethod
    def collate(samples: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collates a list of samples into a padded batch of inputs and targets.
        Each input sample in the list should be a tuple of (input, target) tensors.
        Also returns the lengths of unpadded inputs and targets for use in loss
        functions such as CTC or RNN-T.

        Follows time-first format. That is, the retured batch is of shape (T, N, ...).
        """
        inputs = [sample[0] for sample in samples]  # [(T, ...)]
        targets = [sample[1] for sample in samples]  # [(T,)]

        # Batch of inputs and targets padded along time
        input_batch = nn.utils.rnn.pad_sequence(inputs)  # (T, N, ...)
        target_batch = nn.utils.rnn.pad_sequence(targets)  # (T, N)

        # Lengths of unpadded input and target sequences for each batch entry
        input_lengths = torch.as_tensor([len(_input) for _input in inputs], dtype=torch.int32)
        target_lengths = torch.as_tensor([len(target) for target in targets], dtype=torch.int32)

        return {
            "inputs": input_batch,
            "targets": target_batch,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths,
        }

    @classmethod
    def from_s3(
        cls,
        s3_path: str,
        transform: Transform[np.ndarray, torch.Tensor] = field(default_factory=ToTensor),
        window_length: int | None = None,
        stride: int | None = None,
        padding: tuple[int, int] = (0, 0),
        jitter: bool = False,
        region: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,
        cache_dir: str | None = None,
    ) -> "WindowedEMGDataset":
        """Create a WindowedEMGDataset from an S3 path using s3torchconnector.

        Args:
            s3_path: S3 URI to the HDF5 file (s3://bucket/path/to/file.hdf5)
            transform: Transform to apply to the EMG data
            window_length: Size of each window
            stride: Stride between consecutive windows
            padding: Left and right contextual padding for windows
            jitter: Whether to randomly jitter window offsets
            region: AWS region
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            endpoint_url: S3 endpoint URL
            cache_dir: Directory to cache downloaded files

        Returns:
            A WindowedEMGDataset instance that loads data from S3
        """
        import os
        import tempfile

        from s3torchconnector import S3Checkpoint

        # Configure S3 client
        s3_config = {}
        if aws_access_key_id:
            s3_config["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            s3_config["aws_secret_access_key"] = aws_secret_access_key
        if endpoint_url:
            s3_config["endpoint_url"] = endpoint_url

        # Create S3 checkpoint for downloading
        checkpoint = S3Checkpoint(region=region, **s3_config)

        # Create a temporary file or use cache_dir if provided
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            local_path = os.path.join(cache_dir, os.path.basename(s3_path))
        else:
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, f"emg2qwerty_{os.path.basename(s3_path)}")

        # Download the file if it doesn't exist locally
        if not os.path.exists(local_path):
            with checkpoint.reader(s3_path) as reader, open(local_path, "wb") as f:
                f.write(reader.read())

        # Create a regular WindowedEMGDataset with the local file
        return cls(
            hdf5_path=Path(local_path),  # Convert to Path
            window_length=window_length,
            stride=stride,
            padding=padding,
            jitter=jitter,
            transform=transform,
        )


class S3WindowedEMGDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` that loads EMG data from S3 buckets using s3torchconnector.

    This dataset provides the same functionality as WindowedEMGDataset but loads data
    directly from S3 without saving files locally.

    Args:
        s3_path (str): S3 path to the session file in hdf5 format (s3://bucket/path/to/file.hdf5)
        window_length (int): Size of each window. Specify None for no windowing
        stride (int): Stride between consecutive windows.
        padding (tuple[int, int]): Left and right contextual padding for windows.
        jitter (bool): If True, randomly jitter the offset of each window.
        region (str): AWS region
        aws_access_key_id (str): AWS access key ID
        aws_secret_access_key (str): AWS secret access key
        endpoint_url (str): S3 endpoint URL
        transform (Callable): A transform to apply to each window.
    """

    def __init__(
        self,
        s3_path: str,
        window_length: int | None = None,
        stride: int | None = None,
        padding: tuple[int, int] = (0, 0),
        jitter: bool = False,
        region: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,
        transform: Transform[np.ndarray, torch.Tensor] = field(default_factory=ToTensor),
    ) -> None:
        # S3 configuration
        self.s3_path = s3_path
        self.region = region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url

        # Parse S3 path
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        parts = s3_path.split("/", 1)
        self.bucket = parts[0]
        self.key = parts[1] if len(parts) > 1 else ""

        # Dataset parameters
        self.window_length = window_length
        self.stride = stride if stride is not None else window_length
        self.padding = padding
        self.jitter = jitter
        self.transform = transform

        # Set up S3 client config
        self.s3_config = {}
        if aws_access_key_id and aws_secret_access_key:
            self.s3_config["aws_access_key_id"] = aws_access_key_id
            self.s3_config["aws_secret_access_key"] = aws_secret_access_key
        if endpoint_url:
            self.s3_config["endpoint_url"] = endpoint_url
        S3ClientConfig()

        # Initialize S3Checkpoint for reading data
        self.checkpoint = S3Checkpoint(region=region)

        # Initialize session metadata
        self._initialize_metadata()

        # Set up window parameters
        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        # Cache for frequently accessed data
        self._window_cache: dict[tuple[int, int], np.ndarray] = {}
        self._timestamps_cache: np.ndarray | None = None

    def _initialize_metadata(self) -> None:
        """Initialize dataset metadata by reading only the necessary parts of the HDF5 file."""
        with self.checkpoint.reader(f"s3://{self.bucket}/{self.key}") as reader:
            with h5py.File(reader, "r") as h5file:
                # Get the EMG2qwerty group
                emg2qwerty_group = h5file[EMGSessionData.HDF5_GROUP]

                # Get the timeseries dataset shape without loading all data
                timeseries = emg2qwerty_group[EMGSessionData.TIMESERIES]
                self.session_length = len(timeseries)

                # Load metadata (small enough to keep in memory)
                self.metadata = {}
                for key, val in emg2qwerty_group.attrs.items():
                    if key in {EMGSessionData.KEYSTROKES, EMGSessionData.PROMPTS}:
                        self.metadata[key] = json.loads(val)
                    else:
                        self.metadata[key] = val

                # Store the dataset dtype for later use
                self.timeseries_dtype = timeseries.dtype

                # Cache timestamps if they're not too large (for efficient slicing)
                if self.session_length < 1_000_000:  # Only cache if reasonable size
                    self._timestamps_cache = timeseries[EMGSessionData.TIMESTAMPS][:]
                else:
                    self._timestamps_cache = None

    def __len__(self) -> int:
        """Return the number of windows in the dataset."""
        if self.window_length is None:
            return 1  # Return the entire session
        return int(max(0, (self.session_length - self.window_length) // self.stride + 1))

    def _fetch_window(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Fetch a specific window of data from S3 efficiently."""
        # Check if window is in cache
        cache_key = (start_idx, end_idx)
        if cache_key in self._window_cache:
            return self._window_cache[cache_key]

        # Read data from S3
        with self.checkpoint.reader(f"s3://{self.bucket}/{self.key}") as reader:
            with h5py.File(reader, "r") as h5file:
                emg2qwerty_group = h5file[EMGSessionData.HDF5_GROUP]
                timeseries = emg2qwerty_group[EMGSessionData.TIMESERIES]

                # Read only the specific slice we need
                window_data: np.ndarray = timeseries[start_idx:end_idx]

        # Cache the window if it's small enough (< 1MB)
        if window_data.nbytes < 1_000_000 and len(self._window_cache) < 100:
            self._window_cache[cache_key] = window_data

        return window_data

    def _get_timestamps(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Get timestamps for a specific window, using cache if available."""
        if self._timestamps_cache is not None:
            return self._timestamps_cache[start_idx:end_idx]
        else:
            # Fetch only the timestamps column to minimize data transfer
            with self.checkpoint.reader(f"s3://{self.bucket}/{self.key}") as reader:
                with h5py.File(reader, "r") as h5file:
                    emg2qwerty_group = h5file[EMGSessionData.HDF5_GROUP]
                    timeseries = emg2qwerty_group[EMGSessionData.TIMESERIES]
                    return timeseries[EMGSessionData.TIMESTAMPS][start_idx:end_idx]

    def ground_truth(self, start_t: float, end_t: float) -> LabelData:
        """Get ground truth labels for a specific time window."""
        condition = self.metadata.get(EMGSessionData.CONDITION)

        if condition == "on_keyboard":
            return LabelData.from_keystrokes(
                self.metadata[EMGSessionData.KEYSTROKES], start_t=start_t, end_t=end_t
            )
        else:
            return LabelData.from_prompts(
                self.metadata[EMGSessionData.PROMPTS], start_t=start_t, end_t=end_t
            )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a window of EMG data and corresponding labels."""
        if self.window_length is None:
            # Return the entire session
            window = self._fetch_window(0, self.session_length)
            emg = self.transform(window)

            # Get timestamps for the entire session
            timestamps = self._get_timestamps(0, self.session_length)
            start_t = timestamps[0]
            end_t = timestamps[-1]

            # Get labels for the entire session
            label_data = self.ground_truth(start_t, end_t)
            labels = torch.as_tensor(label_data.labels)

            return emg, labels

        # Calculate window indices with jitter
        offset = idx * self.stride

        # Apply jitter if enabled
        leftover = self.session_length - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Expand window to include contextual padding
        window_start = max(offset - self.left_padding, 0)
        window_end = min(offset + self.window_length + self.right_padding, self.session_length)

        # Fetch the window data
        window = self._fetch_window(window_start, window_end)

        # Apply transform to get tensor
        emg = self.transform(window)
        assert torch.is_tensor(emg)

        # Get timestamps for label alignment
        timestamps = self._get_timestamps(window_start, window_end)
        start_t = timestamps[offset - window_start]
        end_t = timestamps[(offset + self.window_length - 1) - window_start]

        # Get labels for the window
        label_data = self.ground_truth(start_t, end_t)
        labels = torch.as_tensor(label_data.labels)

        return emg, labels
