#!/bin/bash
# Pre-train Potato AI (tiny config for validation)
python -m training.pretrain --config tiny --max_steps 100 --save_every 50
