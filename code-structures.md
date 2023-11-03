
Overview of the code structures used in the project.

Tree structure
--------------
```
.
├── README.md
├── training.py
├── helper.py
├── utils
│   ├── params.py
│   ├── utils.py
├── data
│   ├── MNIST
│   ├── CIFAR-10
│   ├── CIFAR-100
```

Running code:
-------------
```
python training.py --name mnist --params exps/mnist_fed.yaml
```
Flow of the code:
-----------------
1. `training.py` is the main file that is run.
- It parses the arguments and loads the parameters from the yaml file.
- Load all the configurations to `helper.py` (parameters as a variable in a `Helper` class)

Perform the following steps for each round:
- Training `epochs` communcation rounds
- Traing each round for `fl_local_epochs` epochs
- Evaluate the model on the test set


1. Define the `task.py` in the `tasks` folder, that inherits from the `Task` class.
- `Task` class has following functions:
  - `load_data` - load the data from the `data` folder, and split it for different clients.
  - `build_model` - build the model for the task
  - `resume_model` - resume the model from the checkpoint
  - `make_criterion` - define the loss function for the task.
  - `make_optimizer` - define the optimizer for the task.
  - `sample_adversaries` - sample the adversaries for the task.
  - `train` - train the model for one epoch.
  - `metrics` - define the metrics for the task, 2 main metrics are `AccuracyMetric` (top-k metrics) and `TestLossMetric`.
