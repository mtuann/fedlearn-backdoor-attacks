import logging
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from synthesizers.synthesizer import Synthesizer
from attacks.loss_functions import compute_all_losses_and_grads
from utils.parameters import Params
import math
logger = logging.getLogger('logger')


class Attack:
    params: Params
    synthesizer: Synthesizer
    local_dataset: DataLoader
    loss_tasks: List[str]
    fixed_scales: Dict[str, float]
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']

    def __init__(self, params, synthesizer):
        self.params = params
        self.synthesizer = synthesizer
        self.loss_tasks = ['normal', 'backdoor']
        self.fixed_scales = {'normal':0.5, 'backdoor':0.5}

    def perform_attack(self, _) -> None:
        raise NotImplemented

    def compute_blind_loss(self, model, criterion, batch, attack, fixed_model=None):
        """

        :param model:
        :param criterion:
        :param batch:
        :param attack: Do not attack at all. Ignore all the parameters
        :return:
        """
        batch = batch.clip(self.params.clip_batch)
        loss_tasks = self.loss_tasks.copy() if attack else ['normal']
        batch_back = self.synthesizer.make_backdoor_batch(batch, attack=attack)
        scale = dict()

        if len(loss_tasks) == 1:
            loss_values = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back
            )
        else:
            loss_values = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back,
                fixed_model = fixed_model)

            for t in loss_tasks:
                scale[t] = self.fixed_scales[t]

        if len(loss_tasks) == 1:
            scale = {loss_tasks[0]: 1.0}
        blind_loss = self.scale_losses(loss_tasks, loss_values, scale)

        return blind_loss

    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for it, t in enumerate(loss_tasks):
            self.params.running_losses[t].append(loss_values[t].item())
            self.params.running_scales[t].append(scale[t])
            if it == 0:
                blind_loss = scale[t] * loss_values[t]
            else:
                blind_loss += scale[t] * loss_values[t]
        self.params.running_losses['total'].append(blind_loss.item())
        return blind_loss

    def scale_update(self, local_update: Dict[str, torch.Tensor], gamma):
        for name, value in local_update.items():
            value.mul_(gamma)

    def get_fl_update(self, local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            local_update[name] = (data - global_model.state_dict()[name])

        return local_update
    
    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False

    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if 'tracked' in name or 'running' in name:
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm