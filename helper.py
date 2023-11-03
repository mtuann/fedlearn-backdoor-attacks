import importlib
import logging
import os
import random
from shutil import copyfile
from collections import defaultdict

import numpy as np
import torch
import yaml

# from attacks.attack import Attack
# from defenses.fedavg import FedAvg as Defense
# from synthesizers.synthesizer import Synthesizer
# from tasks.task import Task
from utils.parameters import Params
from utils.utils import create_logger
import pandas as pd
logger = logging.getLogger('logger')


class Helper:
    params: Params = None
    # task: Task = None
    # synthesizer: Synthesizer = None
    # defense: Defense = None
    # attack: Attack = None

    def __init__(self, params):
        
        self.params = Params(**params)
        self.times = {'backward': list(), 'forward': list(), 'step': list(),
                      'scales': list(), 'total': list(), 'poison': list()}

        if self.params.random_seed is not None:
            self.fix_random(self.params.random_seed)
            
        self.make_folders()
        
        # self.make_task()
        
        # self.make_synthesizer()
        
        # self.make_attack()
        
        # self.make_defense()
        
        # self.accuracy = [[],[]]
        
        # self.best_loss = float('inf')
        # exit(0)
    def make_task(self):
        name_lower = self.params.task.lower()
        name_cap = self.params.task
        module_name = f'tasks.{name_lower}_task'
        path = f'tasks/{name_lower}_task.py'
        logger.info(f'make task: {module_name} name_cap: {name_cap} path: {path}')
        try:
            task_module = importlib.import_module(module_name)
            task_class = getattr(task_module, f'{name_cap}Task')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your task: {self.params.task} should '
                                      f'be defined as a class '
                                      f'{name_cap}'
                                      f'Task in {path}')
        self.task = task_class(self.params)

    def make_synthesizer(self):
        name_lower = self.params.synthesizer.lower()
        name_cap = self.params.synthesizer
        module_name = f'synthesizers.{name_lower}_synthesizer'
        # logger.info(f'make synthesizer: {module_name} name_cap: {name_cap}')
        try:
            synthesizer_module = importlib.import_module(module_name)
            task_class = getattr(synthesizer_module, f'{name_cap}Synthesizer')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(
                f'The synthesizer: {self.params.synthesizer}'
                f' should be defined as a class '
                f'{name_cap}Synthesizer in '
                f'synthesizers/{name_lower}_synthesizer.py')
        self.synthesizer = task_class(self.task)

    def make_attack(self):
        name_lower = self.params.attack.lower()
        name_cap = self.params.attack
        module_name = f'attacks.{name_lower}'
        logger.info(f'make attack: {module_name} name_cap: {name_cap}')
        try:
            attack_module = importlib.import_module(module_name)
            attack_class = getattr(attack_module, f'{name_cap}')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your attack: {self.params.attack} should '
                                      f'be defined either ThrDFed (3DFed) or \
                                        ModelReplace (Model Replacement Attack)')
        self.attack = attack_class(self.params, self.synthesizer)

    def make_defense(self):
        name_lower = self.params.defense.lower()
        name_cap = self.params.defense
        module_name = f'defenses.{name_lower}'
        try:
            defense_module = importlib.import_module(module_name)
            defense_class = getattr(defense_module, f'{name_cap}')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your defense: {self.params.defense} should '
                                      f'be one of the follow: FLAME, Deepsight, \
                                        Foolsgold, FLDetector, RFLBAT, FedAvg')
        self.defense = defense_class(self.params)

    def make_folders(self):
        log = create_logger()
        if self.params.log:
            os.makedirs(self.params.folder_path, exist_ok=True)

            fh = logging.FileHandler(
                filename=f'{self.params.folder_path}/log.txt')
            formatter = logging.Formatter('%(asctime)s - %(filename)s - Line:%(lineno)d  - %(levelname)-8s - %(message)s')
                        
            fh.setFormatter(formatter)
            log.addHandler(fh)

            with open(f'{self.params.folder_path}/params.yaml.txt', 'w') as f:
                yaml.dump(self.params, f)
        # log.info(f"Creating folder {self.params.folder_path}")
        

    def save_model(self, model=None, epoch=0, val_loss=0):

        if self.params.save_model:
            logger.info(f"Saving model to {self.params.folder_path}.")
            model_name = '{0}/model_last.pt.tar'.format(self.params.folder_path)
            saved_dict = {'state_dict': model.state_dict(),
                          'epoch': epoch,
                          'lr': self.params.lr,
                          'params_dict': self.params.to_dict()}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params.save_on_epochs:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False,
                                     filename=f'{self.params.folder_path}/model_epoch_{epoch}.pt.tar')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}_best')
                self.best_loss = val_loss

    def save_update(self, model=None, userID = 0):
        folderpath = '{0}/saved_updates'.format(self.params.folder_path)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        update_name = '{0}/update_{1}.pth'.format(folderpath, userID)
        torch.save(model, update_name)

    def remove_update(self):
        for i in range(self.params.fl_total_participants):
            file_name = '{0}/saved_updates/update_{1}.pth'.format(self.params.folder_path, i)
            if os.path.exists(file_name):
                os.remove(file_name)
        os.rmdir('{0}/saved_updates'.format(self.params.folder_path))
        if self.params.defense == 'Foolsgold':
            for i in range(self.params.fl_total_participants):
                file_name = '{0}/foolsgold/history_{1}.pth'.format(self.params.folder_path, i)
                if os.path.exists(file_name):
                    os.remove(file_name)
            os.rmdir('{0}/foolsgold'.format(self.params.folder_path))

    def record_accuracy(self, main_acc, backdoor_acc, epoch):
        self.accuracy[0].append(main_acc)
        self.accuracy[1].append(backdoor_acc)
        name = ['main', 'backdoor']
        acc_frame = pd.DataFrame(columns=name, data=zip(*self.accuracy), 
                                    index=range(self.params.start_epoch, epoch+1))
        filepath = f"{self.params.folder_path}/accuracy.csv"
        acc_frame.to_csv(filepath)
        logger.info(f"Saving accuracy record to {filepath}")

    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params.save_model:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def fix_random(seed=1):
        from torch.backends import cudnn

        logger.warning('Setting random_seed seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = False
        cudnn.enabled = True
        cudnn.benchmark = True
        np.random.seed(seed)

        return True
