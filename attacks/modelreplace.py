import torch
# from attack import Attack
from attacks.attack import Attack

class ModelReplace(Attack):

    def __init__(self, params, synthesizer):
        super().__init__(params, synthesizer)
        self.loss_tasks.append('cs_constraint')
        self.fixed_scales = {'normal':0.3, 
                            'backdoor':0.3, 
                            'cs_constraint':0.4}

    def perform_attack(self, _, epoch):
        if self.params.fl_number_of_adversaries <= 0 or \
            epoch not in range(self.params.poison_epoch,\
            self.params.poison_epoch_stop):
            return

        folder_name = f'{self.params.folder_path}/saved_updates'
        file_name = f'{folder_name}/update_0.pth'
        loaded_params = torch.load(file_name)
        self.scale_update(loaded_params, self.params.fl_weight_scale)
        for i in range(self.params.fl_number_of_adversaries):
            file_name = f'{folder_name}/update_{i}.pth'
            torch.save(loaded_params, file_name)