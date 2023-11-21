Here is the list of all the experiments on CIFAR-10 dataset.

```
/home/vishc2/anaconda3/envs/cardio/bin/python training.py --name cifar10 --params exps/cifar_fed.yaml
```
- Exp 01: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 200/ 20/ 4; fl_dirichlet_alpha: 0.5
- Exp 02: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 200/ 10/ 4; fl_dirichlet_alpha: 0.5
- Exp 03: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5
# - Exp 04: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 5/ 4; fl_dirichlet_alpha: 0.5

- Exp 04: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.01
- Exp 05: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.02
- Exp 06: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.05
- Exp 07: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 0; fl_dirichlet_alpha: 0.5; lr: 0.01
- Exp 08: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 0; fl_dirichlet_alpha: 0.5; lr: 0.02
- Exp 09: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 0; fl_dirichlet_alpha: 0.5; lr: 0.05
  

- Exp 10: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.001
- Exp 11: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.002
- Exp 12: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.005