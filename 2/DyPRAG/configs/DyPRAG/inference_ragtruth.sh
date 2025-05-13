
model_name=qwen2.5-1.5b-instruct
python3 src/inference_ragtruth.py \
    --model_name=$model_name \
    --dataset=ragtruth \
    --data_type="QA" \
    --sample=-1 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --inference_method=dyprag_combine \
    --inference_epoch=1 \
    --projector_path="qwen2.5-1.5b-instruct_projector_1stage_hidden32_mse+lm+kl_dataaug_lastlogits_1e-5_sample480*1 " \
    --projector_p=32 \
