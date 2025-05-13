MODELS=(
    "llama3-8b-instruct"
    "llama3.2-1b-instruct"
    "qwen2.5-1.5b-instruct"
)
DATASETS=(
    "complexwebquestions"
    "2wikimultihopqa"
    "hotpotqa"
    "popqa"
)
declare -A WITH_COT
WITH_COT=(
    ["2wikimultihopqa"]=true
    ["complexwebquestions"]=false
    ["hotpotqa"]=true
    ["popqa"]=false
)

SAMPLE=-1
LEARNING_RATE=0.0003
LORA_RANK=2
LORA_ALPHA=32
projector=false
EPOCH=1
BATCH_SIZE=1

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [[ $dataset == "popqa" ]]; then
            EPOCH=2
        fi
        echo "Running encode for model: $model, dataset: $dataset"
        # Build command
        cmd="python3 -u src/encode.py \
            --model_name=$model \
            --dataset=$dataset \
            --sample=$SAMPLE \
            --per_device_train_batch_size=$BATCH_SIZE \
            --num_train_epochs=$EPOCH \
            --learning_rate=$LEARNING_RATE \
            --lora_rank=$LORA_RANK \
            --lora_alpha=$LORA_ALPHA \
            "
        
        if [ "${projector}" = true ]; then
            cmd="$cmd --projector"
        fi

        if [ "${WITH_COT[$dataset]}" = true ]; then
            cmd="$cmd --with_cot"
        fi

        # Execute command and save log
        echo "Executing: $cmd"
        eval "$cmd" 
        
        # Wait for current process to finish before starting next one
        wait
    done
done
