# An active perception framework for BlueRov

## Installation

Tested on Ubuntu 20.04

### Prerequisites

Python >= 3.8

Several gigabytes of storage

pip3

stable-baselines3:master ant its prerequisites

transform3d >=0.0.4

CUDA

HoloOcean

### Installation Instruction

```
pip install stable-baselines3[extra]
pip3 install transform3d
pip install holoocean
```
Use CUDA version corresponding to your pytorch version to use GPU speedup. Strongly recommand to install the newest pytorch version by yourself before use `pip install stable-baselines3[extra]`


## Script Description

### Naive Flight Simulation Environment

All files to create and train the model inside the environment is under document `naive_flight_simulation_environment`

This environment includes a robot equipped with a camera and 3 platforms. Two of them are black and the other is red. We want to let the robot land on the red platform. You can use `python train_flight_model.py` under `naive_flight_simulation_environment` to run the training script. After the training process is finished, the model will be stored under document `naive_flight_simulation_environment`. In terms of more details about this environment, check the 590 report: https://www.overleaf.com/read/nfccvbwxsmpd#bbc6e7.

- `flight_feature_extractor.py`:
  - An extractor used in this environment to process raw data captured from the sensors on robot

- `flight_landing_env.py`:
  - Main code to build the environment

Other files contain helpful function to create the environment. See code to have more details.

### Ocean Environment

This environment is a easy demo to check if we can use stable-baselines3 with HoloOcean. The answer is yes but currently HoloOcean do not have speed up about simulation time which makes the training process unacceptably expensive. As a result, we create the naive flight environment in preivous section. However, code in this section can be a framework to apply in the future once the problem is fixed.

- `ocean_env.py`
  - Main code to build the environment

- `train_ocean_model.py`
  - the script to run the training process. After the training process is finished, the model will be stored under document `Ocean_environment`
  
- `example_to_test_train_model.py`
  - An example code to use the trained model

We also write a PD controller under document `Holo_simultion/controller.py`

