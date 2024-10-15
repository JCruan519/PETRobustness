export CUDA_VISIBLE_DEVICES=0,1


t_mode='adaptformer' # gist_seq_adapter gist_shared_par_adapter adaptformer
model='vit_large_patch16_224_in21k'
model_t='vit_small_patch16_224_in21k'
NODE_NUM=2
DATA_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/data/vtab-1k
exp_tag=''
CSV_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/${t_mode}/${exp_tag}/
CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
OUTPUT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/
master_port=12447
train_py=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/train_onlinekd.py

python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/cifar  \
    --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model $model  --model_t $model_t  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/cifar100/${t_mode} \
	--amp  --tuning-mode $t_mode --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH}   \

# t_mode='adaptformer' # gist_seq_adapter gist_shared_par_adapter adaptformer
# model='swin_large_patch4_window7_224_in22k'
# NODE_NUM=2
# DATA_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/data/vtab-1k
# exp_tag=''
# CSV_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/${t_mode}/${exp_tag}/
# CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
# OUTPUT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/
# master_port=12447
# train_py=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/train.py

# python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
# 	${train_py} ${DATA_ROOT_PATH}/cifar  \
#     --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model $model  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}${model}/vtab/cifar_100/${t_mode} \
# 	--amp  --tuning-mode $t_mode --pretrained  \
#     --csv_root_path ${CSV_ROOT_PATH} \
#     --csv_path ${CSV_PATH} \




# t_mode='adaptformer' # gist_seq_adapter gist_shared_par_adapter adaptformer
# model='swin_base_patch4_window7_224_in22k'
# NODE_NUM=2
# DATA_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/data/vtab-1k
# exp_tag=''
# CSV_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/${t_mode}/${exp_tag}/
# CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
# OUTPUT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/
# master_port=12447
# train_py=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/train.py

# python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
# 	${train_py} ${DATA_ROOT_PATH}/cifar  \
#     --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model $model  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}${model}/vtab/cifar_100/${t_mode} \
# 	--amp  --tuning-mode $t_mode --pretrained  \
#     --csv_root_path ${CSV_ROOT_PATH} \
#     --csv_path ${CSV_PATH} \




# t_mode='adaptformer' # gist_seq_adapter gist_shared_par_adapter adaptformer
# model='swin_small_patch4_window7_224_22k'
# NODE_NUM=2
# DATA_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/data/vtab-1k
# exp_tag=''
# CSV_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/${t_mode}/${exp_tag}/
# CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
# OUTPUT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/
# master_port=12447
# train_py=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/train.py

# python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
# 	${train_py} ${DATA_ROOT_PATH}/cifar  \
#     --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model $model  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}${model}/vtab/cifar_100/${t_mode} \
# 	--amp  --tuning-mode $t_mode --pretrained  \
#     --csv_root_path ${CSV_ROOT_PATH} \
#     --csv_path ${CSV_PATH} \




# t_mode='adaptformer' # gist_seq_adapter gist_shared_par_adapter adaptformer
# model='swin_base_patch4_window7_224_in22k'
# model_t='swin_small_patch4_window7_224_22k'

# NODE_NUM=2
# DATA_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/data/vtab-1k
# exp_tag=''
# CSV_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/${t_mode}/${exp_tag}/
# CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
# OUTPUT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/
# master_port=12447
# train_py=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/train_onlinekd.py


# python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
# 	${train_py} ${DATA_ROOT_PATH}/cifar  \
#     --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model $model  --model_t $model_t  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}${model}/vtab/cifar_100/${t_mode} \
# 	--amp  --tuning-mode $t_mode --pretrained  \
#     --csv_root_path ${CSV_ROOT_PATH} \
#     --csv_path ${CSV_PATH}   \




# t_mode='adaptformer' # gist_seq_adapter gist_shared_par_adapter adaptformer
# model='swin_base_patch4_window7_224_in22k'
# model_t='swin_base_patch4_window7_224_in22k'

# NODE_NUM=2
# DATA_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/data/vtab-1k
# exp_tag=''
# CSV_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/${t_mode}/${exp_tag}/
# CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
# OUTPUT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/
# master_port=12447
# train_py=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/train_onlinekd.py


# python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
# 	${train_py} ${DATA_ROOT_PATH}/cifar  \
#     --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model $model  --model_t $model_t  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}${model}/vtab/cifar_100/${t_mode} \
# 	--amp  --tuning-mode $t_mode --pretrained  \
#     --csv_root_path ${CSV_ROOT_PATH} \
#     --csv_path ${CSV_PATH}   \




# t_mode='adaptformer' # gist_seq_adapter gist_shared_par_adapter adaptformer
# model='swin_base_patch4_window7_224_in22k'
# model_t='swin_large_patch4_window7_224_in22k'

# NODE_NUM=2
# DATA_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/data/vtab-1k
# exp_tag=''
# CSV_ROOT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/${t_mode}/${exp_tag}/
# CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
# OUTPUT_PATH=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/outputs/
# master_port=12447
# train_py=/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_KD/train_onlinekd.py


# python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
# 	${train_py} ${DATA_ROOT_PATH}/cifar  \
#     --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model $model  --model_t $model_t  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}${model}/vtab/cifar_100/${t_mode} \
# 	--amp  --tuning-mode $t_mode --pretrained  \
#     --csv_root_path ${CSV_ROOT_PATH} \
#     --csv_path ${CSV_PATH}   \