#!/bin/bash
# Agent fine-tuning from SFT checkpoint
python -m training.agent_finetune --config tiny --checkpoint checkpoints/sft_tiny.pt --max_steps 30
