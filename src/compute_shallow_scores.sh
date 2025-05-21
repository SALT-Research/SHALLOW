CUDA_DEVICE=1
DATASET_NAME=ls
MODEL_NAME=canary1b

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python main.py \
    --dataset_name $DATASET_NAME \
    --model_name $MODEL_NAME \
    --gt_transcriptions_path ./gt/${DATASET_NAME}_gt.txt \
    --predictions_path ./inference/${DATASET_NAME}/${DATASET_NAME}_${MODEL_NAME}.txt \
    --output_dir ./results/ \
    --examples_limit -1 \
    --num_workers 4