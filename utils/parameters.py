from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict
import logging
import torch
logger = logging.getLogger('logger')

@dataclass
class Params:

    # Corresponds to the class module: tasks.mnist_task.MNISTTask
    # See other tasks in the task folder.
    task: str = 'MNIST'

    current_time: str = None
    name: str = None
    random_seed: int = None
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # training params
    start_epoch: int = 1
    epochs: int = None
    poison_epoch: int = None
    poison_epoch_stop: int = None
    log_interval: int = 1000

    # model arch is usually defined by the task
    resume_model: str = None
    lr: float = None
    decay: float = None
    momentum: float = None
    optimizer: str = None
    # data
    data_path: str = '.data/'
    batch_size: int = 64
    test_batch_size: int = 100
    # Do not apply transformations to the training images.
    transform_train: bool = True
    # For large datasets stop training earlier.
    max_batch_id: int = None
    # No need to set, updated by the Task class.
    input_shape = None

    # attack params
    backdoor: bool = False
    backdoor_label: int = 8
    poisoning_proportion: float = 1.0  # backdoors proportion in backdoor loss
    synthesizer: str = 'pattern'
    backdoor_dynamic_position: bool = False

    # factors to balance losses
    fixed_scales: Dict[str, float] = None

    # optimizations:
    alternating_attack: float = None
    clip_batch: float = None
    # Disable BatchNorm and Dropout
    switch_to_eval: float = None

    # logging
    report_train_loss: bool = True
    log: bool = False
    save_model: bool = None
    save_on_epochs: List[int] = None
    save_scale_values: bool = False
    print_memory_consumption: bool = False
    save_timing: bool = False
    timing_data = None

    # Temporary storage for running values
    running_losses = None
    running_scales = None

    # FL params
    fl: bool = False
    fl_no_models: int = 100
    fl_local_epochs: int = 2
    fl_poison_epochs: int = None
    fl_total_participants: int = 80000
    fl_eta: int = 1
    fl_sample_dirichlet: bool = False
    fl_dirichlet_alpha: float = None
    fl_diff_privacy: bool = False
    # FL attack details. Set no adversaries to perform the attack:
    fl_number_of_adversaries: int = 0
    fl_single_epoch_attack: int = None
    fl_weight_scale: int = 1

    attack: str = None #'ThrDFed' (3DFed), 'ModelRplace' (Model Replacement)
    
    #"Foolsgold", "FLAME", "RFLBAT", "Deepsight", "FLDetector"
    defense: str = None 
    lagrange_step: float = None
    random_neurons: List[int] = None
    noise_mask_alpha: float = None
    fl_adv_group_size: int = 0
    fl_num_neurons: int = 0

    def __post_init__(self):
        # enable logging anyways when saving statistics
        if self.save_model or self.save_timing or \
                self.print_memory_consumption:
            self.log = True

        if self.log:
            self.folder_path = f'saved_models/' \
                               f'{self.task}_{self.current_time}_{self.name}'

        self.running_losses = defaultdict(list)
        self.running_scales = defaultdict(list)
        self.timing_data = defaultdict(list)

    def to_dict(self):
        return asdict(self)