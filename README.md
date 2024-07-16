# CookingScrambledEgg
## Cooking Robot Mastery by Learning Online Motion Generation with Active Perception
The open source cord for running the learning model.

<img src="https://github.com/namikosaito/CookingScrambledEgg/blob/main/Cooking_cover_img.png" align="middle" width="3000"/>


## System Repuirements
Confirmed with the PCs with spec below
* Operating System: Linux(Ubuntu) 
* CPU:
  * Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz
  * AMD Ryzen 7 6800HS with Radeon Graphics
* GPU:
  * Nvidia GeForce RTX 2080 SUPER with CUDA 12.4
  * Nvidia GeForce RTX 3060 Mobile / Max-Q with CUDA 12.2
* Disk Space: Approximately 1.8GB of free space required
Generic computer hardware is sufficient and no non-standard hardware is necessary to run the project.

## Intallation
Use the virtual environment (venv), "cooking_env"

``` $ source cooking_env/bin/activate```

FYI
* python 3.8.10
* pip 24.1.2
* numpy 2.0.0
* opencv-python 4.10.0.84
* matplotlib 3.7.5
* pytorch 2.3.1 (check and follow installattion here: [pytorch link](https://pytorch.org/get-started/locally/))

## How to run CAE
### 0. Introduction
  The aim of CAE is to learn image and extract image features.
  The sample training dataset images(.png) are put in dataset folderm, which are used in our research.
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
  The sample trained models are already put in the result folder, which is used for our evaluation in the paper.
  * The model for trimmed image: result/1023-0500-trimmed/snap/01500.tar
  * The model for wholr image: result/1025-2335-whole/snap/01500.tar
    
You can use your own model after training, too.

  ```$ cd ae```
  
  ```$ python do_cae.py test```

  The image features and reconstructed images will be saved in the result folder (result/MMDD-TIME-trimmed(whole)/snap/).

### 3. Sample reconstruct results

## How to run MTRNN + Attention
### 0. Introduction
  This model learn sequential sensorimotor data (image feature, motor angle, force and tactile sensor) to predict the next step data.

### 1. Training MTRNN + Attention model
  Set the parameters/directry path in "rnn/do_rnn.py"
  * the number of neuron of Cf and Cs nodes (c_size)
  * the time constant for Cs and Cs nodes (tau)
  * open rate (input_param)
    
  etc

   ```$ cd rnn```
   
   ```$ python do_rnn.py train```

  The trained log and models will be saved in the result folder

### 2. Test MTRNN + Attention model
  The sample trained models are already put in the result folder, which is used in our evaluation.
  * result/0716-1843_cf30_cs7_cft5_cst32/snap/20000.tar
    
You can use your own model after training, too.
    
  ```$ cd rnn```
  
  ```$ python do_rnn.py test```

### 3. Sample motion generation (prediction) results
