
Overview of the code structures used in the project.

Tree structure
--------------
```
.
├── README.md
├── training.py
│   ├── __init__.py
├── dataset.py
├── model.py
├── attack_methods.py
├── defense_methods.py
├── method.sh
├── utils
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   ├── plot.py
│   ├── save_load.py
│   └── utils.py
├── data
│   ├── MNIST
│   │   ├── processed
│   │   │   ├── test.pt
│   │   │   └── training.pt
│   │   └── raw
│   │       ├── t10k-images-idx3-ubyte
│   │       ├── t10k-images-idx3-ubyte.gz
│   │       ├── t10k-labels-idx1-ubyte
│   │       ├── t10k-labels-idx1-ubyte.gz
│   │       ├── train-images-idx3-ubyte
│   │       ├── train-images-idx3-ubyte.gz
│   │       ├── train-labels-idx1-ubyte
│   │       └── train-labels-idx1-ubyte.gz
│   ├── CIFAR-10
│   │   ├── processed
│   │   │   ├── test_batch
│   │   │   ├── training_batch_1
│   │   │   ├── training_batch_2
│   │   │   ├── training_batch_3
```

Running code:
-------------
```
python training.py --name mnist --params exps/mnist_fed.yaml
```
