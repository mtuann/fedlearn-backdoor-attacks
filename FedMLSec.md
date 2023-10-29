Table of Contents
-----------------


# Setup Python Environment
- Set the name of environment in both files: `environment.yml` and `Makefile`. The default name is `aba`, aka "all backdoor attacks" and then run following commands:
```
    make install
```

# Guideline for custome training (Atk vs Def)

## Data Customization
- [Data Loader](https://github.com/FedML-AI/FedML/blob/master/doc/en/simulation/user_guide/data_loader_customization.md)

## Datasets and Models Customization
- [Datasets and Models](https://github.com/FedML-AI/FedML/blob/master/doc/en/simulation/user_guide/datasets-and-models.md#datasets-and-models)
- [FedML Data](https://github.com/FedML-AI/FedML/tree/master/python/fedml/data)

## Attack Customization
### [Model Replacement Attack (MRA)](https://arxiv.org/pdf/1807.00459.pdf)
- Code is in `fedlearn-backdoor-attacks/3DFed/attacks/modelreplace.py`
![Alt text](uploaded-figures/model-replacement-attack.png)



## Defense Customization

## Training Customization

## Evaluation Customization

## Visualization Customization

## Result Customization


## Flow of the code:
- Init data, attack, defense method in `fedlearn-backdoor-attacks/3DFed/helper.py`
- Run fl round in `fedlearn-backdoor-attacks/3DFed/training.py`
    - Sample user for round (fl_no_models/ fl_total_participants)
    - Init FLUser (user_id, compromised, train_loader)
        - Single epoch attack: user_id = 0 is attacker, compromised = True
        - Otherwise: check if epoch in attack_epochs, if yes, check list adversaries
    - Training for each user
        - If user is attacker, run attack for only user_id = 0 [missing other attackers], that means training models on poisoned data
        - Otherwise, run defense for all users
    - Perform Attack and aggregate results
        - check update_global_model with weight: 1/total_participants (self.params.fl_eta / self.params.fl_total_participants)
    - Limitation: Currently, dump fl_no_models (set = fl_total_participants) models  in each round into file, only one attacker is supported (other attackers is duplicated from attacker 0)
## TODO:
- [ ] Setting standard FL attack from Attack at the Tail and DBA
- [ ] Change dump to file -> dump to memory
- [ ] Check popular defense method: Foolsgold, RFA, ...

## Reference


# Source
- [FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and Federated LLMs](https://arxiv.org/pdf/2306.04959.pdf)
- [Attack and Defense of FedMLSecurity](https://github.com/FedML-AI/FedML/blob/master/python/fedml/core/security/readme.md)
- [fedMLSecurity_experiments](https://github.com/FedML-AI/FedML/tree/master/python/examples/security/fedMLSecurity_experiments)

