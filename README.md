<!-- Table of contents -->
# Table of contents
- [Table of contents](#table-of-contents)
- [Survey Papers for Machine Learning Security](#survey-papers-for-machine-learning-security)
  - [Paper Backdoor Attack in ML/ FL](#paper-backdoor-attack-in-ml-fl)
  - [Code for Backdoor Attack in ML/ FL](#code-for-backdoor-attack-in-ml-fl)
  - [Other Resources for Backdoor Attack in ML/ FL](#other-resources-for-backdoor-attack-in-ml-fl)


# Description
This github project provided a fast integration for readers to work in the field of backdoor attacks in machine learning and federated learning.

# Overview of project
- Attack methods: DBA, LIRA, BackdoorBox, 3DFed, Chameleon, etc.
- Defense methods: Krum, RFA, FoolsGold, etc.

# How to develop from this project
- Define dataset in `dataset.py` file.
- Define your own attack method in `attack_methods.py` file.
- Define your own defense method in `defense_methods.py` file.
- Define your own model in `model.py` file.
---
Run the following command to start the project:
```
bash method.sh
```
---

# Dataset for Backdoor Attack in FL
<!-- MD Table -->
|Dataset|Case|Description|
| :--- | :--- | :--- |
|MNIST|Case 1|The MNIST database of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples.|


[Edge-case backdoors](https://proceedings.neurips.cc/paper/2020/hash/b8ffa41d4e492f0fad2f13e29e1762eb-Abstract.html): 

# Experiments setting
- Dataset: MNIST, CIFAR-10, CIFAR-100, etc.
- Attack methods: DBA, LIRA, BackdoorBox, 3DFed, Chameleon, etc.
- Defense methods: Krum, RFA, FoolsGold, etc.
- Model: CNN, ResNet, etc.
- Hyperparameters: learning rate, batch size, etc.
<!-- Experiments setting Table -->
|Dataset|Backdoor-case|Attack methods|Defense methods|Model|Hyperparameters|$\mathcal{D}_{edge}$|
|:--- | :--- | :--- | :--- | :--- | :--- | :--- |
|CIFAR-10|Edge-case|-|-|VGG-9|K = 200, m = 10|Southwest Airline’s planes (truck)|
|EMNIST|Edge-case|-|-|LeNet|K = 3383, m = 30|"7"s from Ardis (1) |
|ImageNet (ILSVRC-2012)|Edge-case|-|-|VGG-11|K = 1000, m = 10| certain ethnic dresses (irrelevant label) |


# Working with FedML
- [FedML README.md](https://github.com/FedML-AI/FedML/blob/master/python/fedml/core/security/readme.md)

Types of Attacks in FL setting:
- Byzantine Attack: 
- DLG Attack: 
- Backdoor Attack: 
- Model Replacement Attack: 


# Survey Papers for Machine Learning Security
<!-- MD Table -->
| Title | Year | Venue | Code | Dataset | URL | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|A Survey on Fully Homomorphic Encryption: An Engineering Perspective|2017|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3124441)||
|Generative Adversarial Networks: A Survey Toward Private and Secure Applications|2021|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3459992)||
|A Survey on Adversarial Recommender Systems: From Attack/Defense Strategies to Generative Adversarial Networks|2021|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3439729)||
|Video Generative Adversarial Networks: A Review|2022|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3487891)||
|Taxonomy of Machine Learning Safety: A Survey and Primer|2022|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3551385)||
|Dataset Security for Machine Learning: Data Poisoning, Backdoor Attacks, and Defenses|2022|ACM Computing Surveys|||[link](https://arxiv.org/pdf/2012.10544.pdf)||
|Generative Adversarial Networks: A Survey on Atack and Defense Perspective|2023|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3615336)||
|Trustworthy AI: From Principles to Practices|2023|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3555803)||
|Deep Learning for Android Malware Defenses: A Systematic Literature Review|2022|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3571156)||
|A Comprehensive Review of the State-of-the-Art on Security and Privacy Issues in Healthcare|2023|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3571156)||
|A Comprehensive Survey of Privacy-preserving Federated Learning: A Taxonomy, Review, and Future Directions|2023|ACM Computing Surveys|||[link](https://dl.acm.org/doi/pdf/10.1145/3460427)||





## Paper Backdoor Attack in ML/ FL
- [How to Backdoor Federated Learning](https://arxiv.org/pdf/1807.00459.pdf)

## Code for Backdoor Attack in ML/ FL
<!-- Table for Backdoor Attack in ML/ FL -->
| Title | Year | Venue | Code | Dataset | URL | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|Practicing-Federated-Learning||Github|[link](https://github.com/FederatedAI/Practicing-Federated-Learning/)|
|Attack of the Tails: Yes, You Really Can Backdoor Federated Learning||NeurIPS'20|[link](https://github.com/ksreenivasan/OOD_Federated_Learning)|
|DBA: Distributed Backdoor Attacks against Federated Learning||ICLR'20|[link](https://github.com/AI-secure/DBA)|
|LIRA: Learnable, Imperceptible and Robust Backdoor Attacks ||ICCV'21|[link](https://github.com/khoadoan106/backdoor_attacks/tree/main)|
|Backdoors Framework for Deep Learning and Federated Learning||AISTAT'20, USENIX'21|[link](https://github.com/ebagdasa/backdoors101)|
|BackdoorBox: An Open-sourced Python Toolbox for Backdoor Attacks and Defenses|2023|Github|[link](https://github.com/THUYimingLi/BackdoorBox)|
|3DFed: Adaptive and Extensible Framework for Covert Backdoor Attack in Federated Learning||IEEE S&P'23|[link](https://github.com/haoyangliASTAPLE/3DFed)|
|Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning||ICML'23|[link](https://github.com/ybdai7/Chameleon-durable-backdoor)|



## Other Resources for Backdoor Attack in ML/ FL
- [List of papers on data poisoning and backdoor attacks](https://github.com/penghui-yang/awesome-data-poisoning-and-backdoor-attacks)
- [Proceedings of Machine Learning Research](https://proceedings.mlr.press/)