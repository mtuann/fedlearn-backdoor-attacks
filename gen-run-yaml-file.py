import os
from datetime import datetime


def generate_exps_file(root_file='./exps/cifar_fed.yaml', name_exp = 'cifar10', EXPS_DIR="./exps/extras"):
    # read file as a string
    print(f'reading from root_file: {root_file}')
    with open(root_file, 'r') as file :
        filedata = file.read()
    fl_total_participants_choices = [100, 200]
    fl_no_models_choices = [10, 20]
    fl_dirichlet_alpha_choices = [0.5]
    fl_number_of_adversaries_choices = [4]
    fl_lr_choices = [0.05]
    # EXPS_DIR = './exps/extras'
    
    os.makedirs(EXPS_DIR, exist_ok=True)

    for fl_total_participants in fl_total_participants_choices:
        for fl_no_models in fl_no_models_choices:
            for fl_dirichlet_alpha in fl_dirichlet_alpha_choices:
                for fl_number_of_adversaries in fl_number_of_adversaries_choices:
                    for fl_lr in fl_lr_choices:
                        print(f'fl_total_participants: {fl_total_participants} fl_no_models: {fl_no_models} fl_dirichlet_alpha: {fl_dirichlet_alpha} fl_number_of_adversaries: {fl_number_of_adversaries} fl_lr: {fl_lr}')
                        filedata = filedata.replace('fl_total_participants: 100', f'fl_total_participants: {fl_total_participants}')
                        filedata = filedata.replace('fl_no_models: 10', f'fl_no_models: {fl_no_models}')
                        filedata = filedata.replace('fl_dirichlet_alpha: 0.5', f'fl_dirichlet_alpha: {fl_dirichlet_alpha}')
                        filedata = filedata.replace('fl_number_of_adversaries: 4', f'fl_number_of_adversaries: {fl_number_of_adversaries}')
                        filedata = filedata.replace('lr: 0.005', f'lr: {fl_lr}')
                        # print(len(filedata), type(filedata))  
                        # print('------------------------')
                        # write the file out again
                        fn_write = f'{EXPS_DIR}/{name_exp}_fed_{fl_total_participants}_{fl_no_models}_{fl_number_of_adversaries}_{fl_dirichlet_alpha}_{fl_lr}.yaml'
                        if not os.path.exists(fn_write):
                            with open(fn_write, 'w') as file:
                                file.write(filedata)
                            
                        cmd = f'python training.py --name {name_exp} --params {fn_write}'
                        print(cmd)
                        print('------------------------')

current_time = datetime.now().strftime('%Y.%b.%d')

generate_exps_file(root_file='./exps/cifar_fed.yaml', 
                   name_exp = 'cifar10', 
                   EXPS_DIR=f"./exps/run_cifar10__{current_time}")


generate_exps_file(root_file='./exps/mnist_fed.yaml', 
                   name_exp = 'mnist', 
                   EXPS_DIR=f"./exps/run_mnist__{current_time}")

generate_exps_file(root_file='./exps/imagenet_fed.yaml', 
                   name_exp = 'tiny-imagenet', 
                   EXPS_DIR=f"./exps/run_tiny-imagenet__{current_time}")

