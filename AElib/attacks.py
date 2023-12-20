import torch
import torch.nn.functional as F
import torch.nn as nn

# FGSM Attack
def perturb(imgs, eps, data_grads):
    # Collect the element-wise sign of the data gradient
    sign_data_grads = data_grads.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    adv_imgs = imgs + eps * sign_data_grads
    # Adding clipping to maintain [0,1] range
    adv_imgs = torch.clamp(adv_imgs, 0, 1)
    # Return the perturbed image
    return adv_imgs

def fgsm_attack(model, imgs, labels, eps):
    imgs.required_grad = True
    
    outputs = model(imgs)
    loss = F.nll_loss(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    data_grads = imgs.grad.data
    
    adv_imgs = perturb(imgs, eps, data_grads)
    outputs = model(adv_imgs)
    new_preds = outputs.argmax(axis=1)
    
    return adv_imgs, new_preds

# PGD Attack
def pgd_linf(model, imgs, labels, epsilon, alpha, num_iter, flag=0):
    """ Construct PGD adversarial examples on the examples X"""
    delta = torch.zeros_like(imgs, requires_grad=True)
    if flag == 1:
        num_iter = 1
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(imgs + delta), labels)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
        new_preds = model(imgs + delta).argmax(axis=1)
    return (imgs + delta).detach(), new_preds