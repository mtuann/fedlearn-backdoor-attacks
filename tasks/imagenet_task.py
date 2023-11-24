import random

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import Subset

from models.resnet_tinyimagenet import resnet18
from tasks.task import Task
import os
import logging
logger = logging.getLogger('logger')


class ImagenetTask(Task):

    def load_data(self):
        self.load_imagenet_data()
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            # train_loaders = [self.get_train(indices) for pos, indices in
            #                  indices_per_participant.items()]
            
            train_loaders, number_of_samples = zip(*[self.get_train(indices) for pos, indices in
                            indices_per_participant.items()])
        else:
            # sample indices for participants that are equally
            # split to 500 images per participant
            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            train_loaders = [self.get_train_old(all_range, pos)
                             for pos in
                             range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples
        logger.info(f"Done splitting with #participant: {self.params.fl_total_participants}")
        
        return

    def load_imagenet_data(self):

        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        test_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(self.params.data_path, 'train'),
            train_transform)

        self.test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(self.params.data_path, 'val'),
            test_transform)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True, num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=8, pin_memory=True)

    def build_model(self) -> None:
        return resnet18(pretrained=False)
