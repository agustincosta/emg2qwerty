#!/bin/bash

# Activate virtualenv
echo "Activating virtualenv..."
source .venv/bin/activate

# Run first training
echo "Starting first training..."
python -m emg2qwerty.train model=tds_conv_ctc_tiny checkpoint=checkpoints/model-tiny-new/model-tiny-new-epochepoch_28-val_metricval_metric_0.0000.ckpt model_name=model-tiny-new

# Check if first training completed successfully
if [ $? -eq 0 ]; then
    echo "First training completed successfully. Starting second training..."
    
    # Run second training
    python -m emg2qwerty.train model=tds_conv_ctc_autoencoder_tiny model_name=model-autoencoder-tiny-new
    
    # Check if second training completed successfully
    if [ $? -eq 0 ]; then
        echo "Second training completed successfully. Shutting down..."
    else
        echo "Second training failed."
    fi
else
    echo "First training failed. Not proceeding with second training."
fi 
# Shutdown the system
sudo shutdown -h now