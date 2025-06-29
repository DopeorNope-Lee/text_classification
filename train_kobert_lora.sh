export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=$(pwd):$PYTHONPATH


python src/train.py \
  --model_name "skt/kobert-base-v1" \
  --output_dir models/checkpoints/kobert_lora \
  --num_epochs 5 \
  --batch_size 16 \
  --lr 2e-5 \
  --fp16 \
  --use_lora  