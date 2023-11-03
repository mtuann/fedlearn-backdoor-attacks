import logging
import random
from typing import List, Any, Dict
from copy import deepcopy
import numpy as np
from collections import defaultdict

import torch
from torch import optim, nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from tasks.batch import Batch
from tasks.fl_user import FLUser
from utils.parameters import Params

logger = logging.getLogger('logger')


class Task:
    params: Params = None

    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    classes = None

    model: Module = None
    optimizer: optim.Optimizer = None
    criterion: Module = None
    metrics: List[Metric] = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    "Generic normalization for input data."
    input_shape: torch.Size = None

    fl_train_loaders: List[Any] = None
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']
    adversaries: List[int] = None

    def __init__(self, params: Params):
        self.params = params
        self.init_task()

    def init_task(self):
        self.load_data()
        self.model = self.build_model()
        self.resume_model()
        self.model = self.model.to(self.params.device)

        self.local_model = self.build_model().to(self.params.device)
        self.criterion = self.make_criterion()
        self.adversaries = self.sample_adversaries()

        # self.optimizer = self.make_optimizer()
        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()

        # Initialize the logger
        fh = logging.FileHandler(
                filename=f'{self.params.folder_path}/log.txt')
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def load_data(self) -> None:
        raise NotImplemented

    def build_model(self) -> Module:
        raise NotImplemented

    def make_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        return nn.CrossEntropyLoss(reduction='none')

    def make_optimizer(self, model=None) -> Optimizer:
        if model is None:
            model = self.model
        if self.params.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.params.lr,
                                  weight_decay=self.params.decay,
                                  momentum=self.params.momentum)
        elif self.params.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=self.params.lr,
                                   weight_decay=self.params.decay)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

        return optimizer

    def resume_model(self):
        if self.params.resume_model:
            logger.info(f'Resuming training from {self.params.resume_model}')
            loaded_params = torch.load(f"saved_models/"
                                       f"{self.params.resume_model}",
                                    map_location=torch.device('cpu'))
            self.model.load_state_dict(loaded_params['state_dict'])
            self.params.start_epoch = loaded_params['epoch']
            # self.params.lr = loaded_params.get('lr', self.params.lr)

            logger.warning(f"Loaded parameters from saved model: LR is"
                           f" {self.params.lr} and current epoch is"
                           f" {self.params.start_epoch}")

    def set_input_shape(self):
        inp = self.train_dataset[0][0]
        self.params.input_shape = inp.shape

    def get_batch(self, batch_id, data) -> Batch:
        """Process data into a batch.

        Specific for different datasets and data loaders this method unifies
        the output by returning the object of class Batch.
        :param batch_id: id of the batch
        :param data: object returned by the Loader.
        :return:
        """
        inputs, labels = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

    def accumulate_metrics(self, outputs, labels):
        for metric in self.metrics:
            metric.accumulate_on_batch(outputs, labels)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_metric()

    def report_metrics(self, step, prefix=''):
        metric_text = []
        for metric in self.metrics:
            metric_text.append(str(metric))
        logger.warning(f'{prefix} {step:4d}. {" | ".join(metric_text)}')

        return  self.metrics[0].get_main_metric_value()

    @staticmethod
    def get_batch_accuracy(outputs, labels, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        max_k = max(top_k)
        batch_size = labels.size(0)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        if len(res) == 1:
            res = res[0]
        return res

    def get_empty_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def sample_users_for_round(self, epoch) -> List[FLUser]:
        sampled_ids = random.sample(
            range(self.params.fl_total_participants),
            self.params.fl_no_models)
        sampled_users = []
        for pos, user_id in enumerate(sampled_ids):
            train_loader = self.fl_train_loaders[user_id]
            compromised = self.check_user_compromised(epoch, pos, user_id)
            user = FLUser(user_id, compromised=compromised,
                          train_loader=train_loader)
            sampled_users.append(user)

        return sampled_users

    def check_user_compromised(self, epoch, pos, user_id):
        """Check if the sampled user is compromised for the attack.

        If single_epoch_attack is defined (eg not None) then ignore
        :param epoch:
        :param pos:
        :param user_id:
        :return:
        """
        compromised = False
        if self.params.fl_single_epoch_attack is not None:
            if epoch == self.params.fl_single_epoch_attack:
                # if pos < self.params.fl_number_of_adversaries:
                if user_id == 0:
                    compromised = True
                    logger.warning(f'Attacking once at epoch {epoch}. Compromised'
                                   f' user: {user_id}.')
        else:
            if epoch >= self.params.poison_epoch and epoch < self.params.poison_epoch_stop + 1:
                compromised = user_id in self.adversaries
        return compromised

    def sample_adversaries(self) -> List[int]:
        adversaries_ids = []
        if self.params.fl_number_of_adversaries == 0:
            logger.warning(f'Running vanilla FL, no attack.')
        elif self.params.fl_single_epoch_attack is None:
            adversaries_ids = list(range(self.params.fl_number_of_adversaries))
            logger.warning(f'Attacking over multiple epochs with following '
                           f'users compromised: {adversaries_ids}.')
        else:
            logger.warning(f'Attack only on epoch: '
                           f'{self.params.fl_single_epoch_attack} with '
                           f'{self.params.fl_number_of_adversaries} compromised'
                           f' users.')

        return adversaries_ids

    def get_model_optimizer(self, model):
        local_model = deepcopy(model)
        local_model = local_model.to(self.params.device)

        optimizer = self.make_optimizer(local_model)

        return local_model, optimizer

    def copy_params(self, global_model: Module, local_model: Module):
        local_state = local_model.state_dict()
        for name, param in global_model.state_dict().items():
            if name in local_state and name not in self.ignored_weights:
                local_state[name].copy_(param)

    def update_global_model(self, weight_accumulator, global_model: Module):
        self.last_global_model = deepcopy(self.model)
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            scale = self.params.fl_eta / self.params.fl_total_participants
            average_update = scale * sum_update
            model_weight = global_model.state_dict()[name]
            model_weight.add_(average_update)

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False
    
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: dataset_classes, a preprocessed class-indices dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as
            parameters for
            dirichlet distribution to sample number of images in each class.
        """

        dataset_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in dataset_classes:
                dataset_classes[label].append(ind)
            else:
                dataset_classes[label] = [ind]
        class_size = len(dataset_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(dataset_classes.keys())

        for n in range(no_classes):
            random.shuffle(dataset_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = dataset_classes[n][
                               :min(len(dataset_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                dataset_classes[n] = dataset_classes[n][
                                   min(len(dataset_classes[n]), no_imgs):]

        return per_participant_list

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param indices:
        :return:
        """
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.params.batch_size,
                                  sampler=SubsetRandomSampler(
                                      indices), drop_last=True)
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(
            len(self.train_dataset) / self.params.fl_total_participants)
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.params.batch_size,
                                  sampler=SubsetRandomSampler(
                                      sub_indices))
        return train_loader