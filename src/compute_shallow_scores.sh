CUDA_DEVICE=1
DATASET_NAME=ls
MODEL_NAME=canary1b

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python main.py \
    --dataset_name $DATASET_NAME \
    --model_name $MODEL_NAME \
    --gt_transcriptions_path /home/akoudounas/SHALLOW-old/gt/${DATASET_NAME}_gt.txt \
    --predictions_path /home/akoudounas/SHALLOW-old/inference/${DATASET_NAME}/${DATASET_NAME}_${MODEL_NAME}_zs.txt \
    --output_dir /home/akoudounas/SHALLOW/results/ \
    --examples_limit 50 \
    --num_workers 2