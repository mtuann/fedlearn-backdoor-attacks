import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task
import copy 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from termcolor import colored

from torch.nn.utils import parameters_to_vector, vector_to_parameters


transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()


def get_clip_image(dataset="cifar10"):
    if dataset in ['timagenet', 'tiny-imagenet32', 'Imagenet']:
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset in ['cifar10', 'Cifar10']:
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset in ['mnist', 'MNIST']:
        def clip_image(x):
            return torch.clamp(x, -1.0, 1.0)
    elif dataset == 'gtsrb':
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    else:
        raise Exception(f'Invalid dataset: {dataset}')
    return clip_image

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    # print(f"mask_grad_list_copy: {mask_grad_list_copy}")
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)
            

class IBASynthesizer(Synthesizer):
    
    def __init__(self, task: Task):
        super().__init__(task)
        # self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)
        tgt_model_path = "/home/vishc2/tuannm/fedlearn-backdoor-attacks/synthesizers/checkpoint/lira/lira_cifar10_vgg9_0.03.pt"
        self.atk_model = self.load_adversarial_model(tgt_model_path, self.params.task, self.params.device)

    def get_target_transform(self, target_label, mode="all2one", num_classes=10):
        """
        Get target transform function
        """
        if mode == "all2one":
            target_transform = lambda x: torch.ones_like(x) * target_label
        elif mode == 'all2all':
            target_transform = lambda x: (x + 1) % num_classes            
        else:
            raise Exception(f'Invalid mode {mode}')
        
        return target_transform

    def create_trigger_model(self, dataset, device="cpu", attack_model=None):
        """ Create trigger model for IBA """
        # print(f"device: {device}")
        # /home/vishc2/tuannm/fedlearn-backdoor-attacks/synthesizers
        if dataset in ['cifar10', 'Cifar10']:
            from synthesizers.attack_models.unet import UNet
            atkmodel = UNet(3).to(device)
        elif dataset in ['mnist', 'MNIST']:
            from synthesizers.attack_models.autoencoders import MNISTAutoencoder as Autoencoder
            atkmodel = Autoencoder().to(device)
        elif dataset == 'timagenet' or dataset == 'tiny-imagenet32' or dataset == 'gtsrb' or dataset == 'Imagenet':
            if attack_model is None:
                from synthesizers.attack_models.autoencoders import Autoencoder
                atkmodel = Autoencoder().to(device)
            elif attack_model == 'unet':
                from synthesizers.attack_models.unet import UNet
                atkmodel = UNet(3).to(device)
        else:
            raise Exception(f'Invalid atk model {dataset}')
        return atkmodel

    def load_adversarial_model(self, checkpoint_path, dataset, device):
        # print(f"CKPT file for attack model: {checkpoint_path}")
        atkmodel = self.create_trigger_model(dataset, device)
        atkmodel.load_state_dict(torch.load(checkpoint_path))
        # print(colored("Load attack model sucessfully!", "red"))
        return atkmodel
    
    def get_poison_batch_adversarial(self, bptt, dataset, device, evaluation=False, target_transform=None, atk_model=None, ratio=0.75, atk_eps=0.75):
        images, targets = bptt
        poisoning_per_batch = int(ratio*len(images))
        images = images.to(device)
        targets = targets.to(device)
        poison_count= 0

        clip_image = get_clip_image(dataset)

        with torch.no_grad():
            noise = atk_model(images) * atk_eps
            atkdata = clip_image(images + noise)
            if target_transform:
                atktarget = target_transform(targets)
            atktarget = targets.to(device)
        if not evaluation:
            atkdata = atkdata[:poisoning_per_batch]
            atktarget = atktarget[:poisoning_per_batch]    
        poison_count = len(atkdata)  

        return atkdata.to(device), atktarget.to(device), poison_count
    
    def make_pattern(self, pattern_tensor, x_top, y_top):
        # TRAIN IBA
        # for e in range(self.params.iba_epochs):
            # self.train_iba(self.task.model, self.task.atk_model, self.task.tgt_model, self.task.optimizer, self.task.atkmodel_optimizer, self.task.train_loader, 
            #                self.task.criterion, atkmodel_train=True, device=self.params.device, logger=self.params.logger, 
            #                adv_optimizer=self.task.adv_optimizer, clip_image=get_clip_image(self.params.dataset), target_transform=self.task.target_transform, 
            #                dataset=self.params.dataset, mu=self.params.mu, aggregator=self.params.aggregator, attack_alpha=self.params.attack_alpha, 
            #                attack_portion=self.params.attack_portion, atk_eps=self.params.atk_eps, pgd_attack=self.params.pgd_attack, 
            #                proj=self.params.proj, pgd_eps=self.params.pgd_eps, project_frequency=self.params.project_frequency, 
            #                mask_grad_list=self.task.mask_grad_list, model_original=self.task.model_original, local_e=e)
            # self.task.copy_params(self.task.model, self.task.atk_model)
        # Load model from file and train IBA
        # from torch import optim
        # atk_optimizer = optim.Adam(tgt_model.parameters(), lr=0.00005)
        # tgt_model_path = "/home/vishc2/tuannm/fedlearn-backdoor-attacks/synthesizers/checkpoint/lira/lira_cifar10_vgg9_0.03.pt"
        # target_transform = self.get_target_transform(0, mode="all2one", num_classes=10)
        
        # # task in [Imagenet, Cifar10, MNIST]
        
        # tgt_model = self.load_adversarial_model(tgt_model_path, self.params.task, self.params.device)
        
        # target_transform = self.get_target_transform(self.params.backdoor_label, mode="all2one", num_classes=self.params.num_classes)
        
        # poison_data, poison_target, poison_num = self.get_poison_batch_adversarial(batch_cp, dataset=dataset, device=device,
        #                                                             target_transform=target_transform, atk_model=atk_model)
        
        # self.train_lira(net, tgt_model, None, None, atk_optimizer, local_train_dl,
        #                                     criterion, 0.5, 0.5, 1.0, None, tgt_tf, 1, 
        #                                     atkmodel_train=True, device=self.device, pgd_attack=False)
        
        pass
    def train_iba(self, model, atkmodel, tgtmodel, optimizer, atkmodel_optimizer, train_loader, criterion, atkmodel_train=False, 
                  device=None, logger=None, adv_optimizer=None, clip_image=None, target_transform=None, dataset=None,
                  mu=0.1, aggregator="fedprox", attack_alpha=1.0, attack_portion=1.0, atk_eps=0.1, pgd_attack=False, 
                  proj="l_inf", pgd_eps=0.3, project_frequency=1, mask_grad_list=None, model_original=None, local_e=0):

        wg_clone = copy.deepcopy(model)
        loss_fn = nn.CrossEntropyLoss()
        func_fn = loss_fn
        
        correct_clean = 0
        correct_poison = 0
        
        poison_size = 0
        clean_size = 0
        loss_list = []
        
        if not atkmodel_train:
            model.train()
            # Sub-training phase
            for batch_idx, batch in enumerate(train_loader):
                bs = len(batch)
                data, targets = batch
                # data, target = data.to(device), target.to(device)
                # clean_images, clean_targets, poison_images, poison_targets, poisoning_per_batch = get_poison_batch(batch, attack_portion)
                clean_images, clean_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
                poison_images, poison_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
                # dataset_size += len(data)   
                clean_size += len(clean_images)
                optimizer.zero_grad()
                if pgd_attack:
                    adv_optimizer.zero_grad()
                output = model(clean_images)
                loss_clean = loss_fn(output, clean_targets)
                
                if attack_alpha == 1.0:
                    optimizer.zero_grad()
                    loss_clean.backward()
                    if not pgd_attack:
                        optimizer.step()
                    else:
                        if proj == "l_inf":
                            w = list(model.parameters())
                            # adversarial learning rate
                            eta = 0.001
                            for i in range(len(w)):
                                # uncomment below line to restrict proj to some layers
                                if True:#i == 6 or i == 8 or i == 10 or i == 0 or i == 18:
                                    w[i].data = w[i].data - eta * w[i].grad.data
                                    # projection step
                                    m1 = torch.lt(torch.sub(w[i], model_original[i]), -pgd_eps)
                                    m2 = torch.gt(torch.sub(w[i], model_original[i]), pgd_eps)
                                    w1 = (model_original[i] - pgd_eps) * m1
                                    w2 = (model_original[i] + pgd_eps) * m2
                                    w3 = (w[i]) * (~(m1+m2))
                                    wf = w1+w2+w3
                                    w[i].data = wf.data
                        else:
                            # do l2_projection
                            adv_optimizer.step()
                            w = list(model.parameters())
                            w_vec = parameters_to_vector(w)
                            model_original_vec = parameters_to_vector(model_original)
                            # make sure you project on last iteration otherwise, high LR pushes you really far
                            # Start
                            if (batch_idx%project_frequency == 0 or batch_idx == len(train_loader)-1) and (torch.norm(w_vec - model_original_vec) > pgd_eps):
                                # project back into norm ball
                                w_proj_vec = pgd_eps*(w_vec - model_original_vec)/torch.norm(
                                        w_vec-model_original_vec) + model_original_vec
                                # plug w_proj back into model
                                vector_to_parameters(w_proj_vec, w)

                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    loss_list.append(loss_clean.item())
                    correct_clean += pred.eq(clean_targets.data.view_as(pred)).cpu().sum().item()
                else:
                    if attack_alpha < 1.0:
                        poison_size += len(poison_images)
                        # poison_images, poison_targets = poison_images.to(device), poison_targets.to(device)
                        with torch.no_grad():
                            noise = tgtmodel(poison_images) * atk_eps
                            atkdata = clip_image(poison_images + noise)
                            atktarget = target_transform(poison_targets)
                            # atkdata.requires_grad_(False)
                            # atktarget.requires_grad_(False)
                            if attack_portion < 1.0:
                                atkdata = atkdata[:int(attack_portion*bs)]
                                atktarget = atktarget[:int(attack_portion*bs)]
                        atkoutput = model(atkdata.detach())
                        loss_poison = F.cross_entropy(atkoutput, atktarget.detach())
                    else:
                        loss_poison = torch.tensor(0.0).to(device)
                    loss2 = loss_clean * attack_alpha  + (1.0 - attack_alpha) * loss_poison
                   
                    optimizer.zero_grad()
                    loss2.backward()
                    if mask_grad_list:
                        apply_grad_mask(model, mask_grad_list)
                    if not pgd_attack:
                        optimizer.step()
                    else:
                        if proj == "l_inf":
                            w = list(model.parameters())
                            # adversarial learning rate
                            eta = 0.001
                            for i in range(len(w)):
                                # uncomment below line to restrict proj to some layers
                                if True:#i == 6 or i == 8 or i == 10 or i == 0 or i == 18:
                                    w[i].data = w[i].data - eta * w[i].grad.data
                                    # projection step
                                    m1 = torch.lt(torch.sub(w[i], model_original[i]), -pgd_eps)
                                    m2 = torch.gt(torch.sub(w[i], model_original[i]), pgd_eps)
                                    w1 = (model_original[i] - pgd_eps) * m1
                                    w2 = (model_original[i] + pgd_eps) * m2
                                    w3 = (w[i]) * (~(m1+m2))
                                    wf = w1+w2+w3
                                    w[i].data = wf.data
                        else:
                            # do l2_projection
                            adv_optimizer.step()
                            w = list(model.parameters())
                            w_vec = parameters_to_vector(w)
                            model_original_vec = parameters_to_vector(list(model_original.parameters()))
                            # make sure you project on last iteration otherwise, high LR pushes you really far
                            if (local_e%project_frequency == 0 and batch_idx == len(train_loader)-1) and (torch.norm(w_vec - model_original_vec) > pgd_eps):
                                # project back into norm ball
                                w_proj_vec = pgd_eps*(w_vec - model_original_vec)/torch.norm(
                                        w_vec-model_original_vec) + model_original_vec
                                
                               
                                # plug w_proj back into model
                                vector_to_parameters(w_proj_vec, w)

                    loss_list.append(loss2.item())
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    poison_pred = atkoutput.data.max(1)[1]  # get the index of the max log-probability
                    
                    # correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                    correct_clean += pred.eq(clean_targets.data.view_as(pred)).cpu().sum().item()
                    correct_poison += poison_pred.eq(atktarget.data.view_as(poison_pred)).cpu().sum().item()
                    
        else:
            model.eval()
            # atk_optimizer = optim.Adam(atkmodel.parameters(), lr=0.0002)
            atkmodel.train()
            # optimizer.zero_grad()
            for batch_idx, (batch) in enumerate(train_loader):
                batch_cp = copy.deepcopy(batch)
                data, target = batch
                # print(f"len(clean_data): {len(clean_data)}")
                data, target = data.to(device), target.to(device)
                bs = data.size(0)
                atkdata, atktarget, poison_num = get_poison_batch_adversarial_updated(batch_cp, dataset=dataset, device=device,
                                                                        target_transform=target_transform, atk_model=atkmodel)
                # dataset_size += len(data)
                poison_size += poison_num
                
                ###############################
                #### Update the classifier ####
                ###############################
                atkoutput = model(atkdata)
                loss_p = func_fn(atkoutput, atktarget)
                loss2 = loss_p
                
                atkmodel_optimizer.zero_grad()
                loss2.backward()
                atkmodel_optimizer.step()
                pred = atkoutput.data.max(1)[1]  # get the index of the max log-probability
                correct_poison += pred.eq(atktarget.data.view_as(pred)).cpu().sum().item()
                loss_list.append(loss2.item())
            
        # acc = 100.0 * (float(correct) / float(dataset_size))
        clean_acc = 100.0 * (float(correct_clean)/float(clean_size)) if clean_size else 0.0
        poison_acc = 100.0 * (float(correct_poison)/float(poison_size)) if poison_size else 0.0
        # poison_acc = 100.0 - poison_acc
        
        training_avg_loss = sum(loss_list)/len(loss_list)
        # training_avg_loss = 0.0
        if atkmodel_train:
            logger.info(colored("Training loss = {:.2f}, acc = {:.2f} of atk model this epoch".format(training_avg_loss, poison_acc), "yellow"))
        else:
            logger.info(colored("Training loss = {:.2f}, acc = {:.2f} of cls model this epoch".format(training_avg_loss, clean_acc), "yellow"))
            logger.info("Training clean_acc is {:.2f}, poison_acc = {:.2f}".format(clean_acc, poison_acc))
        del wg_clone    
        

    def synthesize_inputs(self, batch, attack_portion=128):
        # TODO something
        
        # batch.inputs[:attack_portion] = batch.inputs[:attack_portion]
        # from torch import optim
        # atk_optimizer = optim.Adam(tgt_model.parameters(), lr=0.00005)
        
        
        
        # target_transform = self.get_target_transform(self.params.backdoor_label, mode="all2one", num_classes=self.params.num_classes)
        
        
        images = batch.inputs[:attack_portion]
        
        
        device = self.params.device
        images = images.to(device)
        
        clip_image = get_clip_image(self.params.task)
        atk_eps = 0.75
        
        with torch.no_grad():
            noise = self.atk_model(images) * atk_eps
            atkdata = clip_image(images + noise)
             

        batch.inputs[:attack_portion] = atkdata.to(device)
        

    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)
