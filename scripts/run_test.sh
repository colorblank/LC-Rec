

DATASET=Games
DATA_PATH=./data
CKPT_PATH=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/xxx.json

python test.py \
    --gpu_id 0 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 20 \
    --test_prompt_ids all \
    --index_file .index.json
