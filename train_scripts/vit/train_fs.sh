t_mode=['adaptformer','gist2']
# seq_adapter shared_par_adapter adaptformer linear_probe prompt lora gistb gistp gistip gist2
# l1_reg l2_reg
t_coeff=4
GIST_FACTOR=0.25
GISTB_T=0
GIST_LEN=1
GIST_DROPOUT=0.9
model='vit_base_patch16_224_in21k'

if [[ $t_mode == *"gist"* ]]; then
    exp_tag=fs_${t_mode}_${t_coeff}_${GIST_FACTOR}_${GISTB_T}_${GIST_LEN}_${GIST_DROPOUT}
else
    exp_tag=fs_${t_mode}_${t_coeff}
fi

DATA_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/data/FGVC
CSV_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_ALL/outputs/${exp_tag}/all_csv
CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
OUTPUT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_ALL/outputs/${exp_tag}/
master_port=$(($RANDOM + 30000))
train_py=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_ALL/train.py

DATASET_LIST=(food-101 oxford_pets stanford_cars oxford_flowers fgvc_aircraft)
NUM_CLASS_LIST=(101 37 196 102 100)
NODE_NUM=2


for ((i = 0; i < ${#DATASET_LIST[@]}; i++))
do
    for SHOT in 1 2 4 8 16
    do
        for SEED in 0 1 2
        do 
            python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
            ${train_py} ${DATA_ROOT_PATH}/${DATASET_LIST[$i]}  \
            --dataset ${DATASET_LIST[$i]} --num-classes ${NUM_CLASS_LIST[$i]} --direct-resize  --model $model  \
            --batch-size 32 --epochs 100 \
            --opt adamw  --weight-decay 1e-3 \
            --warmup-lr 1e-7 --warmup-epochs 10  \
            --lr 1e-3 --min-lr 1e-8 \
            --drop-path 0.1 --img-size 224 \
            --mixup 0 --cutmix 0 --smoothing 0 --color-jitter 0.4 --aa rand-m9-mstd0.5 \
            --output  ${OUTPUT_PATH}${model}/FVGC-${DATASET_LIST[$i]}-${t_mode}/${SHOT}/${SEED} \
            --amp  --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
            --csv_root_path ${CSV_ROOT_PATH} \
            --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \
            --few_shot_shot ${SHOT} --few_shot_seed ${SEED} --is_few_shot \

        done
    done
done
