import torch
import timm
from torch import nn
from torch.optim import Adam


def fgsm_attack(model, images, labels, epsilon=1/255):
    # 将输入和模型设置到CUDA上
    images, labels = images.to('cuda'), labels.to('cuda')
    model.to('cuda')
    model.eval()  # 设置模型为评估模式

    # 启用图像的梯度计算
    images.requires_grad = True

    # 前向传播以计算损失
    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)

    # 反向传播获取梯度
    model.zero_grad()
    loss.backward()

    # 使用梯度的符号生成对抗样本
    perturbations = epsilon * images.grad.sign()
    perturbed_images = images + perturbations

    # 保持数据在[0,1]的范围内
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images, perturbations

# def fgsm_attack(model, images, labels, eps=1/255, loss_func=None, device="cuda"):

#     images = images.clone().detach().to(device)
#     labels = labels.clone().detach().to(device)
#     if loss_func is None:
#         loss_func = nn.CrossEntropyLoss()

#     images.requires_grad = True
#     outputs = model(images)

#     cost = loss_func(outputs, labels).to(device)

#     # Update adversarial images
#     grad = torch.autograd.grad(cost, images,
#                                retain_graph=False, create_graph=False)[0]

#     adv_images = images + eps*grad.sign()
#     adv_images = torch.clamp(adv_images, min=0, max=1).detach()
#     return adv_images


def pgd_attack(model, images, labels, epsilon=1/255, alpha=0.5/255, iters=5):
    images, labels = images.to('cuda'), labels.to('cuda')
    model.to('cuda')
    model.eval()

    original_images = images.clone().detach()

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            images += alpha * images.grad.sign()
            eta = torch.clamp(images - original_images, min=-epsilon, max=epsilon)
            images = torch.clamp(original_images + eta, min=0, max=1)

        images = images.detach()
        images.requires_grad = True  # 保证下一次循环可以计算梯度
    perturbations = images - original_images
    return images, perturbations

# def pgd_update(images, ori_images, pgd_type, pgd_alpha, pgd_eps, n, momentum=0):
#     grad = images.grad + momentum

#     if pgd_type == 'l_inf':
#         adv_images = images.detach() + pgd_alpha * grad.sign()
#         eta = torch.clamp(adv_images - ori_images, min=-pgd_eps, max=pgd_eps)  # Projection
#     elif pgd_type == 'l_2':
#         gradnorms = grad.reshape(n, -1).norm(dim=1, p=2).view(n, 1, 1, 1)
#         adv_images = images.detach() + pgd_alpha * grad / gradnorms
#         eta = adv_images - ori_images
#         etanorms = eta.reshape(n, -1).norm(dim=1, p=2).view(n, 1, 1, 1)
#         eta = eta * torch.minimum(torch.tensor(1.), pgd_eps / etanorms)  # Projection

#     images = torch.clamp(ori_images + eta, min=0, max=1).detach()
#     return images, grad


# def pgd_attack(
#     model, images, labels, pgd_eps, pgd_alpha, pgd_iters, pgd_type='l_inf',
#     mom_decay=0, random_start=False, device='cuda'
# ):
#     images, labels = images.to('cuda'), labels.to('cuda')
#     model.to('cuda')
#     assert pgd_type in ['l_inf', 'l_2']
#     if device is None:
#         device = images.device
#     model.eval()

#     ori_images = images.clone().detach()
#     n, c, h, w = tuple(images.shape)
#     if random_start:
#         unit_noise = torch.empty_like(images).uniform_(-1, 1)
#         images = images + unit_noise * pgd_eps
#         images = torch.clamp(images, min=0, max=1).detach()

#     momentum = 0  # Initial gradient momentum
#     for _ in range(pgd_iters):
#         model.zero_grad()
#         images = images.clone().detach().requires_grad_(True)

#         if hasattr(model, 'std_model'):  # model is a mixed classifier
#             _, logits_diffable, _ = model(images, return_all=True)
#         else:  # model is a conventional standalone classifier
#             logits_diffable = model(images)

#         if 'mps' in str(device):
#             logits_diffable = logits_diffable.float().to(device)
#         loss = torch.nn.functional.cross_entropy(logits_diffable, labels).to(device)
#         loss.backward()
#         images, momentum = pgd_update(
#             images, ori_images, pgd_type, pgd_alpha, pgd_eps, n, momentum * mom_decay
#         )

#     return images





class CustomViT(nn.Module):
    def __init__(self, model):
        super().__init__()
        # 使用预训练的ViT模型
        self.model = model

    def forward(self, x):
        x = self.model(x)  # 得到特征
        attn_weights = self.model.blocks[-1].attn.attn  # 最后一个block的注意力权重
        return {'logits': x, 'attn_weights': attn_weights}

def calculate_attn_loss(outputs, patch_indices):
    attn_weights = outputs['attn_weights']
    # 计算所有选中块的平均注意力损失
    attn_loss = -attn_weights[:, :, patch_indices, :].mean()
    return attn_loss

def patch_fool_attack(model, images, labels, num_patches=1, max_iter=250, alpha=0.002, eta=0.2, decay_rate=0.95, decay_step=10):
    device = 'cuda'
    model.to(device)
    images = images.to(device)
    labels = labels.to(device)

    total_patches = 196  # 假设图像被分成196个块
    perturbation = torch.zeros_like(images, requires_grad=True)
    optimizer = Adam([perturbation], lr=eta)

    for i in range(max_iter):
        optimizer.zero_grad()
        if (i + 1) % decay_step == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate

        adv_images = images + perturbation
        outputs = model(adv_images)
        ce_loss = nn.CrossEntropyLoss()(outputs['logits'], labels)

        # 随机选择多个块
        patch_indices = torch.randint(low=0, high=total_patches, size=(num_patches,))
        attn_loss = calculate_attn_loss(outputs, patch_indices)

        total_loss = ce_loss + alpha * attn_loss
        total_loss.backward()
        optimizer.step()

    return images + perturbation.detach()