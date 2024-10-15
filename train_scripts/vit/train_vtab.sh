t_mode=['prompt']
# seq_adapter shared_par_adapter adaptformer linear_probe prompt lora gistb gistp gistip gist2
# l1_reg l2_reg
t_coeff=200
GIST_FACTOR=0.75
GISTB_T=0
GIST_LEN=1
GIST_DROPOUT=0.98
model='vit_base_patch16_224_in21k'

if [[ $t_mode == *"gist"* ]]; then
    exp_tag=${t_mode}_${t_coeff}_${GIST_FACTOR}_${GISTB_T}_${GIST_LEN}_${GIST_DROPOUT}
else
    exp_tag=${t_mode}_${t_coeff}
fi

DATA_ROOT_PATH=/cluster/home/gaoxian/DTL/vtab-1k
CSV_ROOT_PATH=/cluster/home/gaoxian/GIST_ALL/outputs/${exp_tag}/all_csv
CSV_PATH=${CSV_ROOT_PATH}/all_csv.csv
OUTPUT_PATH=/cluster/home/gaoxian/GIST_ALL/outputs/${exp_tag}/
master_port=$(($RANDOM + 30000))
train_py=/cluster/home/gaoxian/GIST_ALL/train.py


NODE_NUM=2
BATCH_SIZE=32
export CUDA_VISIBLE_DEVICES=4,5


python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/caltech101  \
    --dataset caltech101 --num-classes 102  --no-aug  --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/caltech101/${t_mode} \
	--amp  --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/cifar  \
    --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/cifar100/${t_mode} \
	--amp  --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/clevr_count \
    --dataset clevr_count --num-classes 8  --no-aug  --direct-resize --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/clevr_count/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/clevr_dist  \
    --dataset clevr_dist --num-classes 6  --no-aug --direct-resize --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-2 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/clevr_dist/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/diabetic_retinopathy  \
    --dataset diabetic_retinopathy --num-classes 5  --no-aug --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/diabetic_retinopathy/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/dmlab  \
    --dataset dmlab --num-classes 6  --no-aug  --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/dmlab/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/dsprites_loc  \
    --dataset dsprites_loc --num-classes 16  --no-aug  --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/dsprites_loc/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/dsprites_ori  \
    --dataset dsprites_ori --num-classes 16  --no-aug   --direct-resize   --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/dsprites_ori/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/dtd  \
    --dataset dtd --num-classes 47  --no-aug --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
    --output  ${OUTPUT_PATH}${model}/vtab/dtd/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/eurosat  \
    --dataset eurosat --num-classes 10  --no-aug  --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 3e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/eurosat/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/oxford_flowers102 \
    --dataset flowers102 --num-classes 102  --no-aug --direct-resize --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/flowers102/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/kitti  \
    --dataset kitti --num-classes 4  --no-aug --direct-resize --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/kitti/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/patch_camelyon  \
    --dataset patch_camelyon --num-classes 2  --no-aug  --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/patch_camelyon/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/oxford_iiit_pet  \
    --dataset pets --num-classes 37  --no-aug --direct-resize --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/pets/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/resisc45  \
    --dataset resisc45 --num-classes 45  --no-aug  --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/resisc45/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/smallnorb_azi  \
    --dataset smallnorb_azi --num-classes 18  --no-aug --direct-resize --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 2e-2 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/smallnorb_azi/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/smallnorb_ele  \
    --dataset smallnorb_ele --num-classes 9  --no-aug  --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/smallnorb_ele/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/sun397  \
    --dataset sun397 --num-classes 397  --no-aug --direct-resize  --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/sun397/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \

 python  -m torch.distributed.launch --nproc_per_node=$NODE_NUM  --master_port=$master_port  \
	${train_py} ${DATA_ROOT_PATH}/svhn  \
    --dataset svhn --num-classes 10  --no-aug --direct-resize --model $model  \
    --batch-size ${BATCH_SIZE} --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}${model}/vtab/svhn/${t_mode} \
	--amp --tuning-mode $t_mode --tuning_coeff ${t_coeff} --pretrained  \
    --csv_root_path ${CSV_ROOT_PATH} \
    --csv_path ${CSV_PATH} --gist_factor ${GIST_FACTOR} --gistb_T ${GISTB_T} --gist_len ${GIST_LEN} --gist_drop ${GIST_DROPOUT} \