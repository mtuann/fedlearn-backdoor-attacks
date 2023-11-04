from tasks.batch import Batch
from tasks.task import Task
from utils.parameters import Params


class Synthesizer:
    params: Params
    task: Task

    def __init__(self, task: Task):
        self.task = task
        self.params = task.params

    def make_backdoor_batch(self, batch: Batch, test=False, attack=True) -> Batch:

        # Don't attack if only normal loss task.
        if not attack:
            return batch

        if test:
            attack_portion = batch.batch_size
        else:
            attack_portion = round(
                batch.batch_size * self.params.poisoning_proportion)

        backdoored_batch = batch.clone()
        self.apply_backdoor(backdoored_batch, attack_portion)
        return backdoored_batch
    
        ### Plot the backdoored image and the original image
        # import IPython; IPython.embed(); exit(0)
        # batch.inputs.shape = (batch_size, 3, 32, 32) = (64, 3,  32, 32)
        # using torch to show the image
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import torchvision
        # import torchvision.transforms as transforms
        # def imshow(img):
        #     # img to cpu
        #     img = img.cpu()
        #     # img = img / 2 + 0.5     # unnormalize
        #     npimg = img.numpy()
        #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #     plt.show()
            
        # def imshow2(img):
        #     # img to cpu
        #     img = img.cpu()
        #     img = img / 2 + 0.5     # unnormalize
        #     npimg = img.numpy()
        #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #     plt.show()
            
            
        # imshow(torchvision.utils.make_grid(batch.inputs))
        # imshow(torchvision.utils.make_grid(backdoored_batch.inputs))
        # import IPython; IPython.embed(); exit(0)
        
        

    def apply_backdoor(self, batch, attack_portion):
        """
        Modifies only a portion of the batch (represents batch poisoning).

        :param batch:
        :return:
        """
        self.synthesize_inputs(batch=batch, attack_portion=attack_portion)
        self.synthesize_labels(batch=batch, attack_portion=attack_portion)

        return

    def synthesize_inputs(self, batch, attack_portion=None):
        raise NotImplemented

    def synthesize_labels(self, batch, attack_portion=None):
        raise NotImplemented
