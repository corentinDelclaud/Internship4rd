MODEL_NAME=(qwen2.5-1.5b-instruct,llama3.2-1b-instruct,llama3-8b-instruct)
dataset_list=(hotpotqa popqa 2wikimultihopqa complexwebquestions)
for model in ${MODEL_NAME[@]}; do
    for dataset in ${dataset_list[@]}; do
        python -u src/augment.py \
        --model_name ${model} \
        --dataset ${dataset} \
        --data_path data/${dataset}/ \
        --sample 200  \
        --topk 3 \
        --output_dir data_aug_projector \
        --projector
    done
done