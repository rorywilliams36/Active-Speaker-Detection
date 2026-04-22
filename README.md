# Lightweight Active-Speaker-Detection using Visual Information
Dissertation project by Rory Williams and Supervised by Dr Yoshi Gotoh 

Aim of the project is to develop an Active-Speaker Detection application that only uses visual data for classification and is lightweight.  

The detector uses Optical Flow to measure the movement in the speaker's face by comparing the current frame to previous frames then a lightweight model is used to classify the value.
  
Unfortunately there are issues with the features as the models are unreliable at differentiating between speakers and non-speakers due to the feature values being too similar therefore, it generally can correctly identify speakers 55\% of the time

### Preparing the Dataset 
Dataset used is the [AVA-ActiveSpeaker Dataset](https://arxiv.org/abs/1901.01342) by Roth et al   

The dataset can be downloaded [here](https://github.com/okankop/ASDNet) or [here](https://research.google.com/ava/download.html)
  
To prepare, in the dataset folder have a train and test folder each containing other folders named after the video ids that contain frames from that video as images with the title of `videoID_timstamp.jpg` (e.g `_mAfwH6i90E_906.0.jpg`)  
The frames are stored in folders named after the video id e.g (`_mAfwH6i90E/_mAfwH6i90E_906.0.jpg`)  

The full dataset layout:   
├── dataset/ # Stored datasets   
│ ├── train/ # stores frames of training video ids (formatted as above)  
│ ├── test/ # Stores frame of testing video ids  
│ └── ava_activespeaker_train_v1.0/ # Stores csv files containing the labels for training data   
│ └── ava_activespeaker_test_v1.0/ # Stores csv files containing the labels for testing data   

This can be done by executing the `/utils/data_prep.py` file by altering the video id input at the function call at the bottom of the file, you will also have to set the boolean to show whether it is training or testing.  
Alternatively, this can be done using the ffmpeg command line package. 

The videos are prepared at 10fps and 100-250 frames are used for each video  
Videos used:  
`train_ids = ['_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g',
'20TAGRElvfE', 'Db19rWN5BGo', 'rFgb2ECMcrY', 'N0Dt9i9IUNg', '8aMv-ZGD4ic', 'Ekwy7wzLfjc', 
'0f39OWEqJ24']`   
 
`test_ids = ['4ZpjKfu6Cl8', '2qQs3Y9OJX0', 'HV0H6oc4Kvs', 'rJKeqfTlAeY', '1j20qq1JyX4', 'C25wkwAMB-w']`



### Usage

First clone repo and install requirements: `pip install -r requirements.txt`   

#### Training:
`python main.py --{train, validate} --{SVM, MobileNet, ShuffleNet}`  
Validate will evaluate the model on the validation set after each training epoch (only to be used with MobileNet or ShuffleNet)  
There are additionally more arguements to change epochs, learning rate and to display loss functions and cross-validation loss

There is option to save the feature data using `--save_features` and loading it again using `--load_features` to save time during training.  
You may have to create an empty folder `feature_data/` in the main project directory to avoid errors.   
These options are only available when training
  
#### Testing: 
`python main.py --test --{SVM, MobileNet, ShuffleNet} {--confMatrix} {--roc}`  

Additionally you can save results using the relevant arguement
 
### Relevant Files/Folders

- `features.py`: Contains code for feature/face extraction from frames and organsises the data with the corresponding labels
- `asd.py`: Code for calculating optical flow values for each feature
- `evaluation.py`: Calculations for all evaluation metrics and also functions for graph creations

`/models`
- `train_vectors.py` all code for training and testing ShuffleNet and MobileNet models
- `support_vec.py` code for training and testing the SVM
- This folder also contains `/parameter_files` folder that contains all parameters for the models, you may have to create this folder to avoid errors.

Other files:  
- Juptyer Notebook `face_eval.ipynb` is used to perform evaluation and experimentation on the face detector
- `/utils` folder contains other useful tools for creating graphs, dataset preparation and some face detection methods   

### Liscense
This project is under the MIT Liscense