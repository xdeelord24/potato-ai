#!/bin/bash
# SFT from pre-trained checkpoint
python -m training.sft --config tiny --checkpoint checkpoints/pretrain_tiny_final.pt --max_steps 50
