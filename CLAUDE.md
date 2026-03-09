# FCON Project - Claude Instructions

## CRITICAL: AWS Credential Refresh (MANDATORY)

**AWS credentials expire every 30 minutes. You MUST maintain a running credential refresh process for the entire duration of any long-running job.**

### Credential refresh command:
```bash
ada credentials update --provider=conduit --account=195966524180 --role=IibsAdminAccess-DO-NOT-DELETE --once
```

### At the start of EVERY session or long-running task:

1. **Start the background refresh loop immediately:**
   ```bash
   /home/fanwanf/lab42/fcon/refresh_creds.sh > /tmp/creds_refresh.log 2>&1 &
   echo $! > /tmp/creds_refresh.pid
   ```

2. **Verify it is running:**
   ```bash
   cat /tmp/creds_refresh.log
   ```

3. **The script refreshes every 25 minutes** (5-minute safety margin before the 30-minute expiry).

4. **If credentials appear stale or a job fails with auth errors**, re-run the refresh command manually and restart the loop.

5. **Check if the loop is still alive** during long jobs:
   ```bash
   kill -0 $(cat /tmp/creds_refresh.pid) 2>/dev/null && echo "running" || echo "DEAD - restart it!"
   ```

The refresh script lives at `/home/fanwanf/lab42/fcon/refresh_creds.sh`.

---

## Project Overview

- **Model**: F-CON (Frustum Convolutional Occupancy Network) for 3D shape completion
- **Paper**: "Convolutional Occupancy Models for Dense Packing of Complex, Novel Objects" (IROS 2023)
- **Dataset**: COB-3D-v2 at `/tmp/cob3d/v2`
- **Checkpoint**: `checkpoints/fcon.pt`

## Environment

- **Conda env**: `fcon` at `/home/fanwanf/miniconda3/envs/fcon`
- **Python**: `/home/fanwanf/miniconda3/envs/fcon/bin/python`
- **GPUs**: 8x NVIDIA L4 (23 GB each)

## Running Evaluation

```bash
# Single GPU
/home/fanwanf/miniconda3/envs/fcon/bin/python evaluate.py \
    --data_root /tmp/cob3d/v2 --checkpoint checkpoints/fcon.pt

# Multi-GPU (recommended)
/home/fanwanf/miniconda3/envs/fcon/bin/python evaluate.py \
    --data_root /tmp/cob3d/v2 --checkpoint checkpoints/fcon.pt --n_workers 8
```

Logs go to `/tmp/fcon_eval.log` when run in background.

## Expected Results (Table I of paper)

| Metric   | Paper value |
|----------|-------------|
| CD-L1    | 0.4229      |
| CD-L2    | 0.1157      |
| F1@0.1   | 0.4664      |
| F1@0.3   | 0.8928      |
| F1@0.5   | 0.9600      |
| Box-IoU  | 0.6809      |
| Box-IoG  | 0.7686      |
| Box-F1   | 0.7485      |
