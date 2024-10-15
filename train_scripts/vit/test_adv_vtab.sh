t_mode=['linear_probe']
# seq_adapter shared_par_adapter adaptformer linear_probe prompt lora gistb gistp gistip gist2
# l1_reg l2_reg
t_coeff=0
GIST_FACTOR=0.75
GISTB_T=0
GIST_LEN=1
GIST_DROPOUT=0.9
model='vit_base_patch16_224_in21k'

if [[ $t_mode == *"gist"* ]]; then
    exp_tag=${t_mode}_${t_coeff}_${GIST_FACTOR}_${GISTB_T}_${GIST_LEN}_${GIST_DROPOUT}
else
    exp_tag=${t_mode}_${t_coeff}
fi

DATA_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/data/vtab-1k
RESUME_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_ALL/outputs_adv/[linear_probe]_0 # TO */vit_base_patch16_224_in21k/vtab
ADV_FILE_NAME="['linear_probe']_0_adv_fgsm"
CSV_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_ALL/outputs_adv/${exp_tag}_${ADV_FILE_NAME}/all_csv
CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
OUTPUT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_ALL/outputs_adv/${exp_tag}_${ADV_FILE_NAME}/
master_port=$(($RANDOM + 30000))
train_py=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_ALL/test_adv.py


NODE_NUM=1
BATCH_SIZE=32
export CUDA_VISIBLE_DEVICES=0

data_path_names=("caltech101" "cifar" "clevr_count" "clevr_dist" "diabetic_retinopathy" "dmlab" "dsprites_loc" "dsprites_ori" "dtd" "eurosat" "oxford_flowers102" "kitti" "patch_camelyon" "oxford_iiit_pet" "resisc45" "smallnorb_azi" "smallnorb_ele" "sun397" "svhn")
dataset_names=("caltech101" "cifar100" "clevr_count" "clevr_dist" "diabetic_retinopathy" "dmlab" "dsprites_loc" "dsprites_ori" "dtd" "eurosat" "flowers102" "kitti" "patch_camelyon" "pets" "resisc45" "smallnorb_azi" "smallnorb_ele" "sun397" "svhn")
data_classes=(102 100 8 6 5 6 16 16 47 10 102 4 2 37 45 18 9 397 10)

array_length=${#data_path_names[@]}

for (( i=0; i<${array_length}; i++ )); do
    echo Dataset Path: ${data_path_names[$i]}, Dataset Name: ${dataset_names[$i]}, Classes: ${data_classes[$i]}

    python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
        ${train_py} ${DATA_ROOT_PATH}/${data_path_names[$i]}  \
        --dataset ${dataset_names[$i]} --num-classes ${data_classes[$i]}  --no-aug  --direct-resize  --model $model  \
        --batch-size ${BATCH_SIZE} --epochs 100 \
        --opt adamw  --weight-decay 5e-2 \
        --warmup-lr 1e-7 --warmup-epochs 10  \
        --lr 1e-3 --min-lr 1e-8 \
        --drop-path 0.1 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
        --output  ${OUTPUT_PATH}${model}/vtab/${dataset_names[$i]}/${t_mode} \
        --amp  --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
        --csv_root_path ${CSV_ROOT_PATH} \
        --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \
        --evaluate --resume ${RESUME_ROOT_PATH}/${dataset_names[$i]}/model_best.pth.tar --adv_file_name ${ADV_FILE_NAME} \
        --is_need_clean

done