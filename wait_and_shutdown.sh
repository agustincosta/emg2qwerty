#!/bin/bash
while ps -p 14477 > /dev/null && ps -p 14990 > /dev/null && ps -p 16175 > /dev/null; do
  echo "Training still running... waiting"
  sleep 300  # Check every 5 minutes
done
echo "Previous training finished, shutting down..."
sudo shutdown now