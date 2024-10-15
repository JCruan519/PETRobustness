import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from tqdm import tqdm
import torch.nn.functional as F
import shutil  # 导入文件操作库
from timm.models import create_model
from attack_methods import fgsm_attack, pgd_attack

###################################### need to be modified
import sys
sys.path.append('/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_ALL/')
from models import vision_transformer
######################################

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = []
        self.img_dir = img_dir
        self.transform = transform
        self.original_size = {}  # 存储原始图像尺寸
        with open(annotations_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.img_labels.append((path, int(label)))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(img_path).convert('RGB')
        self.original_size[idx] = image.size  # 存储原始图像尺寸
        if self.transform:
            image = self.transform(image)
        return image, self.img_labels[idx][1], idx  # 返回原始尺寸的索引
    

def load_model_for_dataset(model_name, tuning_mode, tuning_coeff, dataset_name, num_classes, weight_root_dir=None):
    # 加载针对特定数据集微调后的模型权重
    weight_path = os.path.join(weight_root_dir, f'{dataset_name}/model_best.pth.tar')
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        scriptable=True,
        checkpoint_path=weight_path,
        tuning_mode=tuning_mode,
        tuning_coeff=tuning_coeff)
    return model.cuda()



def save_perturbation(perturbation, original_size):
    # 扰动归一化到 [0, 1]
    perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
    # 转换为 PIL 图像并调整大小
    perturbation_pil = transforms.ToPILImage()(perturbation).resize(original_size)
    return perturbation_pil

def generate_adversarial_samples(source_root_dir, weight_root_dir, data_path_names, data_weights_names,
                                 dataset_classes, txt_files, adv_file_name, model_name, tuning_mode, tuning_coeff, 
                                 transform, attck_method, attack_settings):
    for i_data_name, dataset_name in enumerate(data_path_names):
        num_classes = dataset_classes[i_data_name]
        data_weight_name = data_weights_names[i_data_name]
        print(dataset_name)
        model = load_model_for_dataset(model_name, tuning_mode, tuning_coeff, data_weight_name, num_classes, weight_root_dir)
        model.eval()
        for txt_file in txt_files:
            print(txt_file)
            annotations_file = os.path.join(source_root_dir, dataset_name, txt_file)
            img_dir = os.path.join(source_root_dir, dataset_name)
            save_dir = os.path.join(source_root_dir, dataset_name, 'images', adv_file_name)
            perturbation_dir = os.path.join(save_dir, "perturbations")  # 扰动保存位置
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(perturbation_dir, exist_ok=True)

            dataset = CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, transform=transform)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
            for images, labels, idxs in tqdm(dataloader):
                if attck_method == 'fgsm':
                    adv_images, perturbations = fgsm_attack(model, images, labels, epsilon=attack_settings['eps'])
                elif attck_method == 'pgd':
                    adv_images, perturbations = pgd_attack(model, images, labels, epsilon=attack_settings['eps'], alpha=attack_settings['alpha'], iters=attack_settings['iters'])
                adv_images = adv_images.cpu()
                perturbations = perturbations.cpu()

                for img, perturbation, idx in zip(adv_images, perturbations, idxs):
                    original_size = dataset.original_size[idx.item()]
                    img_pil = transforms.ToPILImage()(img).resize(original_size)
                    perturbation_pil = save_perturbation(perturbation, original_size)
                    
                    img_path = dataset.img_labels[idx.item()][0]
                    save_path = os.path.join(save_dir, os.path.basename(img_path))
                    perturbation_path = os.path.join(perturbation_dir, os.path.basename(img_path))
                    
                    img_pil.save(save_path)
                    perturbation_pil.save(perturbation_path)

    print("完成对抗样本及其扰动的生成和保存。")



# 实际使用
if __name__ == '__main__':

    import time

    start_time = time.time()  # 获取开始时间
    # 放置你的代码

    source_root_dir = '/media/ruanjiacheng/新加卷/ecodes/Prompt/data/vtab-1k'
    data_path_names=("caltech101", "cifar", "clevr_count", "clevr_dist", 
                     "diabetic_retinopathy", "dmlab", "dsprites_loc", "dsprites_ori", 
                     "dtd", "eurosat", "oxford_flowers102", "kitti", "patch_camelyon", 
                     "oxford_iiit_pet", "resisc45", "smallnorb_azi", "smallnorb_ele", "sun397", "svhn")
    data_weights_names=("caltech101", "cifar100", "clevr_count", "clevr_dist", 
                     "diabetic_retinopathy", "dmlab", "dsprites_loc", "dsprites_ori", 
                     "dtd", "eurosat", "flowers102", "kitti", "patch_camelyon", 
                     "pets", "resisc45", "smallnorb_azi", "smallnorb_ele", "sun397", "svhn")
    dataset_classes=(102, 100, 8, 6, 5, 6, 16, 16, 47, 10, 102, 4, 2, 37, 45, 18, 9, 397, 10)
    # txt_files = ['test.txt', 'train800.txt', 'train800val200.txt', 'val200.txt']
    txt_files = ['test_adv_500.txt']

    ######################################### need to be modified
    model_name = 'vit_base_patch16_224_in21k'
    tuning_mode = ['linear_probe']
    tuning_coeff=0
    weight_root_dir='/media/ruanjiacheng/新加卷/ecodes/Prompt/CV/GIST_ALL/outputs_adv/[linear_probe]_0'
    #########################################

    # 图像转换，无变化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    attck_method='pgd'
    attack_settings = {
        'eps': 0.01,
        'alpha': 0.005,
        'iters': 5,
    }
    adv_file_name = f"{tuning_mode}_{tuning_coeff}_adv_{attck_method}"

    generate_adversarial_samples(source_root_dir, 
                                weight_root_dir,
                                data_path_names, 
                                data_weights_names,
                                dataset_classes, 
                                txt_files, 
                                adv_file_name, 
                                model_name, 
                                tuning_mode, 
                                tuning_coeff, 
                                transform, 
                                attck_method,
                                attack_settings)
    

    attck_method='fgsm'
    attack_settings = {
        'eps': 0.01,
    }
    adv_file_name = f"{tuning_mode}_{tuning_coeff}_adv_{attck_method}"

    generate_adversarial_samples(source_root_dir, 
                                weight_root_dir,
                                data_path_names, 
                                data_weights_names,
                                dataset_classes, 
                                txt_files, 
                                adv_file_name, 
                                model_name, 
                                tuning_mode, 
                                tuning_coeff, 
                                transform, 
                                attck_method,
                                attack_settings)
    
    end_time = time.time()  # 获取结束时间
    print(f"执行时间：{end_time - start_time} 秒")