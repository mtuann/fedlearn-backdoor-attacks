import math
from typing import List, Any, Dict
import torch
import logging
import os
from utils.parameters import Params

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FedAvg:
    params: Params
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']

    def __init__(self, params: Params) -> None:
        self.params = params

    # FedAvg aggregation
    def aggr(self, weight_accumulator, _):
        logger.info(f"Aggregating {len(self.params.fl_round_participants)} participants")
        
        weight_contrib = self.params.fl_weight_contribution
        
        for idRound, userID in enumerate(self.params.fl_round_participants):
            updates_name = '{0}/saved_updates/update_{1}.pth'\
                .format(self.params.folder_path, userID)
            # logger.info(f"Aggregating participant {userID} path: {updates_name}")
            
            loaded_params = torch.load(updates_name)
            self.accumulate_weights(weight_accumulator, \
                {key:(loaded_params[key] * weight_contrib[idRound] ).to(self.params.device) for \
                    key in loaded_params})

    def accumulate_weights(self, weight_accumulator, local_update):
        for name, value in local_update.items():
            weight_accumulator[name].add_(value)
    
    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if 'tracked' in name or 'running' in name:
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    def add_noise(self, sum_update_tensor: torch.Tensor, sigma):
        noised_layer = torch.FloatTensor(sum_update_tensor.shape)
        noised_layer = noised_layer.to(self.params.device)
        noised_layer.normal_(mean=0, std=sigma)
        sum_update_tensor.add_(noised_layer)

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False