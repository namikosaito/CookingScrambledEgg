# CookingScrambledEgg
## Cooking Robot Mastery by Learning Online Motion Generation with Active Perception
The open source cord for running the learning model.

<img src="https://github.com/namikosaito/CookingScrambledEgg/blob/main/Cooking_cover_img.png" align="middle" width="3000"/>


## System Repuirements
Confirmed with the PCs with spec below
* Operating System: Linux(Ubuntu) / MacOS Sonoma
* CPU: Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz / Intel Core i5 @ 2GHz 
* GPU: Nvidia GeForce RTX 2080 SUPER with CUDA 11.8 / Intel Iris Plus Graphics
* Disk Space: Approximately XGB of free space required
Generic computer hardware is sufficient and no non-standard hardware is necessary to run the project.

## Intallation
Use the virtual environment (venv), "cooking_env"

* python 3.8.10
* pip 24.1.2
* numpy 2.0.0
* opencv-python 4.10.0.84
* matplotlib 3.7.5
* pytorch 2.3.1 (check and follow installattion here: [pytorch link](https://pytorch.org/get-started/locally/))

## How to run CAE
### 0. Introduction
  The aim of CAE is to learn image and extract image features.
  The sample images(.png) are put in dataset folder.
  * ae/do_cae.py : main program to run
  * ae/model/cae_trimmed.py & cae_whole.py : CAE models
  * ae/src/cae_learn.py : code for training, save trained model in result folder
  * ae/src/cae_eval.py : code for test, extract image fieature, and reconstruct the images

### 1. training CAE model
  Set the parameters/directry path in "ae/do_cae.py"
  * Select Learning_Target (: "trimmed" or "whole") according to the size of the image.
  * GPU number
  * batch size
  etc

   ```$ cd ae```
   
   ```$ python do_cae.py train```

  The trained log and models will be saved in the result folder

### 2. test CAE model
  The sample trained models are already put in the result folder

  ```$ cs ae```
  
  ```$ python do_cae.py test```

  The image features and reconstructed images will be saved in the result folder

### 3. Sample reconstruct results

## How to run MTRNN + Attention
### 0. Introduction

### 1. Training RNN model

### 2. Test RNN model

### 3. Sample results
