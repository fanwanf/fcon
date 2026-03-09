#!/bin/bash
# Refresh AWS credentials via ada every 25 minutes (safety margin before 30-min expiry)
while true; do
    ada credentials update --provider=conduit --account=195966524180 --role=IibsAdminAccess-DO-NOT-DELETE --once
    echo "[$(date)] AWS credentials refreshed"
    sleep 1500  # 25 minutes
done
