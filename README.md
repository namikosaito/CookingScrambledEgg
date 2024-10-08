# CookingScrambledEgg
## Cooking Robot Mastery by Learning Online Motion Generation with Active Perception
The open source cord for running the learning model.

<img src="https://github.com/namikosaito/CookingScrambledEgg/blob/main/Cooking_cover_img.png" align="middle" width="3000"/>


## System Repuirements
Confirmed with the PCs with spec below
* Operating System: Linux(Ubuntu22.04) 
* CPU:
  * Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz
  * AMD Ryzen 7 6800HS with Radeon Graphics
* GPU:
  * Nvidia GeForce RTX 2080 SUPER with CUDA 12.4
  * Nvidia GeForce RTX 3060 Mobile / Max-Q with CUDA 12.2
* Disk Space: Approximately 1.8GB of free space required
  
Generic computer hardware is sufficient and no non-standard hardware is necessary to run the project.

## Intallation

install required dependencies
``` $ pip install -r requirements.txt```


Requirement 
* python >= 3.8
* numpy >= 2.0
* opencv-python >= 4.10
* matplotlib >= 3.7
* pytorch >= 2.3.1 (check and follow installattion here: [pytorch link](https://pytorch.org/get-started/locally/))


FYI
You can use the virtual environment (venv), "cooking_env"

``` $ source cooking_env/bin/activate```


Does not take time for the instalation.

## Dataset
  In this github repository, we put 
  * "dataset/"
    * "raw" dataset of image, motor angle, force and tactile.
    * We only include single data each for training and test due to limited uploading size.
    * This is used for learning CAE
  * "pickle data/"
    * The combined data of motor angle, **image feature**, force and tactile with **all the sequences**
    * This is used for learning MTRNN+Attention   
    
  You can test the codes with the samples above however, if you wish to use the whole "raw" dataset, all the image, motor angle, force and tactile dataset is shared in the drive bellow.
  The size is 21.7GB.
  [Google Drive Link](https://drive.google.com/drive/folders/1SlX5em_k6F9o0GpOT6Nd4VuJyXpfmssN?usp=sharing) 

## How to run CAE
### 0. Introduction
  The aim of CAE is to learn image and extract image features.
  Some part of the sample training dataset images(.png) are put in dataset folderm, which are used in our research.
  * ae/do_cae.py : main program to run
  * ae/model/cae_trimmed.py & cae_whole.py : CAE model architectures
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
  The sample result is saved in "result/" directory
  * image features: result/MMDD-TIME-trimmed(or whole)/snap/XXXXX_mid
  * reconstructed images: result/MMDD-TIME-trimmed(or whole)/snap/XXXXX_rec
  
  The bellow gifs are resonstructed whole & trimmed images.

  <img src="https://github.com/namikosaito/CookingScrambledEgg/blob/main/result/1025-2335-whole/snap/01500_rec/train_normal_4_1/whole.gif"  width="300"/>    <img src="https://github.com/namikosaito/CookingScrambledEgg/blob/main/result/1023-0500-trimmed/snap/01500_rec/train_normal_4_1/trimmed.gif" width="300"/>
<br>

## How to run MTRNN + Attention
### 0. Introduction
  This model learn sequential sensorimotor data (image feature, motor angle, force and tactile sensor) to predict the next step data.
  
  * rnn/do_rnn.py : main program to run
  * rnn/model/mtrnn_attention.py : MTRNN + Attention model architecture
  * rnn/src/rnn_learn.py : code for training, save trained model in result folder
  * rnn/src/rnn_eval.py : code for test, predict the motor, force and tactile data and output with csv and png

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
  The sample result is put in "result/" directry.
  predicted motor angle, force and tactile is output with png and csv.
  * result/MMDD-TIME-cfX-csX-cftX-cstX/snap/XXXXX_seq

  The bellow images are predicted motor angle, force and tactile results of test data (test_blue_4_1).
  In the images, 
  * dotted lines = offline test data (ground truth)
  * solid lines = predicted result by the learning model

  motor angle
  
  <img src="https://github.com/namikosaito/CookingScrambledEgg/blob/main/result/0716-1843_cf30_cs7_cft5_cst32/snap/20000_seq/test_angle_test_blue_4_1.png"  width="500"/>  
  
  force
  
  <img src="https://github.com/namikosaito/CookingScrambledEgg/blob/main/result/0716-1843_cf30_cs7_cft5_cst32/snap/20000_seq/test_force_test_blue_4_1.png" width="500"/>    

  tactile
  
  <img src="https://github.com/namikosaito/CookingScrambledEgg/blob/main/result/0716-1843_cf30_cs7_cft5_cst32/snap/20000_seq/test_tactile_test_blue_4_1.png" width="500"/>
  
