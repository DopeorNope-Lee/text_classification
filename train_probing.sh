export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):$PYTHONPATH

python src/train.py \
  --model_name gpt2 \
  --linear_probing \
  --output_dir models/checkpoints/gpt2_probe_lora \
  --num_epochs 5 \
  --batch_size 16 \
  --lr 5e-4 \
  --fp16
