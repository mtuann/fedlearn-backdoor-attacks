import time

import torch
from torch.nn import functional as F, Module

from models.model import Model
from utils.parameters import Params
from utils.utils import record_time

def compute_all_losses_and_grads(loss_tasks, attack, model, criterion,
                                 batch, batch_back,
                                 fixed_model=None):
    loss_values = {}
    for t in loss_tasks:
        if t == 'normal':
            loss_values[t] = compute_normal_loss(attack.params,
                                                model,
                                                criterion,
                                                batch.inputs,
                                                batch.labels)
        elif t == 'backdoor':
            loss_values[t] = compute_backdoor_loss(attack.params,
                                                    model,
                                                    criterion,
                                                    batch_back.inputs,
                                                    batch_back.labels)
        elif t == 'eu_constraint':
            loss_values[t] = compute_euclidean_loss(attack.params,
                                                    model,
                                                    fixed_model)
        elif t == 'cs_constraint':
            loss_values[t] = compute_cos_sim_loss(attack.params,
                                                    model,
                                                    fixed_model)

    return loss_values


def compute_normal_loss(params: Params, model, criterion, inputs, labels):
    t = time.perf_counter()
    outputs = model(inputs)
    record_time(params, t, 'forward')
    loss = criterion(outputs, labels)
    loss = loss.mean()

    return loss

def compute_backdoor_loss(params, model, criterion, inputs_back, labels_back):
    t = time.perf_counter()
    outputs = model(inputs_back)
    record_time(params, t, 'forward')
    loss = criterion(outputs, labels_back)
    loss = loss.mean()

    return loss

def compute_euclidean_loss(params: Params,
                            model: Model,
                            fixed_model: Model):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.cuda.FloatTensor(size).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (layer - \
            fixed_model.state_dict()[name]).view(-1)
        size += layer.view(-1).shape[0]

    loss = torch.norm(sum_var, p=2)

    return loss

def get_one_vec(model: Module):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

def compute_cos_sim_loss(params: Params,
                            model: Model,
                            fixed_model: Model):
        model_vec = get_one_vec(model)
        target_var = get_one_vec(fixed_model)
        cs_sim = F.cosine_similarity(params.fl_weight_scale*(model_vec-target_var)\
             + target_var, target_var, dim=0)
        loss = 1e3 * (1 - cs_sim)
        return loss

def compute_noise_ups_loss(params: Params, 
                            backdoor_update,
                            noise_masks, 
                            random_neurons):
    losses = []
    for i in range(len(noise_masks)):
        UPs = []
        for j in random_neurons:
            if 'MNIST' not in params.task:
                UPs.append(torch.abs(backdoor_update['fc.weight'][j] + \
                    noise_masks[i].fc.weight[j]).sum() \
                    + torch.abs(backdoor_update['fc.bias'][j] + \
                    noise_masks[i].fc.bias[j]))
            else:
                UPs.append(torch.abs(backdoor_update['fc2.weight'][j] + \
                    noise_masks[i].fc2.weight[j]).sum() \
                    + torch.abs(backdoor_update['fc2.bias'][j] + \
                    noise_masks[i].fc2.bias[j]))
        UPs_loss = 0
        for j in range(len(UPs)):
            if 'Imagenet' in params.task:
                UPs_loss += 5e-4 / UPs[j]
            else:
                UPs_loss += 1e-1 / UPs[j] # (UPs[j] * params.fl_num_neurons)
        noise_masks[i].requires_grad_(True)
        UPs_loss.requires_grad_(True)
        losses.append(UPs_loss)
    return losses

def compute_noise_norm_loss(params: Params,
                        noise_masks,
                        random_neurons):
    size = 0
    layer_name = 'fc2' if 'MNIST' in params.task else 'fc'
    for name, layer in noise_masks[0].named_parameters():
        if layer_name in name:
            size += layer.view(-1).shape[0]
    losses = []
    for i in range(len(noise_masks)):
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        noise_size = 0
        for name, layer in noise_masks[i].named_parameters():
            if layer_name in name:
                for j in range(layer.shape[0]):
                    if j in random_neurons:
                        sum_var[noise_size:noise_size + layer[j].view(-1).shape[0]] = \
                            layer[j].view(-1)
                    noise_size += layer[j].view(-1).shape[0]
        if 'MNIST' in params.task:
            loss = 8e-2 * torch.norm(sum_var, p=2)
        else:
            loss = 3e-2 * torch.norm(sum_var, p=2)
        losses.append(loss)
    return losses

def compute_lagrange_loss(params: Params, 
                            noise_masks, 
                            random_neurons):
    losses = []
    size = 0
    layer_name = 'fc2' if 'MNIST' in params.task else 'fc'
    for name, layer in noise_masks[0].named_parameters():
        if layer_name in name:
            size += layer.view(-1).shape[0]
    sum_var = torch.cuda.FloatTensor(size).fill_(0)
    for i in range(len(noise_masks)):
        size = 0
        for name, layer in noise_masks[i].named_parameters():
            if layer_name in name:
                for j in range(layer.shape[0]):
                    if j in random_neurons:
                        sum_var[size:size + layer[j].view(-1).shape[0]] += \
                            layer[j].view(-1)
                    size += layer[j].view(-1).shape[0]
    
    if 'MNIST' in params.task:
        loss = 1e-1 * torch.norm(sum_var, p=2)
    else:
        loss = 1e-2 * torch.norm(sum_var, p=2)
    for i in range(len(noise_masks)):
        losses.append(loss)
    return losses

def compute_decoy_acc_loss(params: Params, 
                        benign_model: Model, 
                        decoy: Model,
                        criterion, inputs, labels):
    dec_acc_loss, _ = compute_normal_loss(params, decoy, criterion, \
        inputs, labels)
    benign_acc_loss, _ = compute_normal_loss(params, benign_model, criterion, \
        inputs, labels)
    if dec_acc_loss > benign_acc_loss:
        loss = dec_acc_loss
    else:
        loss = - 1e-10 * (dec_acc_loss)

    return loss

def compute_decoy_param_loss(params:Params,
                        decoy: Model,
                        benign_model: Model,
                        param_idx):
    
    if 'MNIST' not in params.task:
        param_diff = torch.abs(decoy.fc.weight[param_idx[0]][param_idx[1]] - \
            benign_model.state_dict()['fc.weight'][param_idx[0]][param_idx[1]])
    else:
        param_diff = torch.abs(decoy.fc1.weight[param_idx[0]][param_idx[1]] - \
            benign_model.state_dict()['fc1.weight'][param_idx[0]][param_idx[1]])

    threshold = 10 # 30
    if(param_diff.item() > threshold):
        loss = 1e-10 * param_diff
    else:
        loss = - 1e1 * param_diff

    return loss

def get_grads(params, model, loss):
    t = time.perf_counter()
    grads = list(torch.autograd.grad(loss.mean(),
                                     [x for x in model.parameters() if
                                      x.requires_grad],
                                     retain_graph=True, allow_unused=True))
    record_time(params, t, 'backward')

    return grads