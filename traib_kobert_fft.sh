export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):$PYTHONPATH

python src/train.py \
  --no_use_lora \
  --model_name "skt/kobert-base-v1" \
  --output_dir models/checkpoints/kobert_full \
  --num_epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --fp16
