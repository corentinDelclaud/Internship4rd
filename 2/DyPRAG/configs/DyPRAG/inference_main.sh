
model_name=Qwen/Qwen2.5-1.5B-Instruct
python3 src/inference_dyprag.py \
    --model_name=$model_name \
    --dataset=2wikimultihopqa \
    --sample=10 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --inference_method=dyprag \
    --inference_epoch=1 \
    --projector_path="projector_path" \
    --projector_p=32 \
    --with_cot \
