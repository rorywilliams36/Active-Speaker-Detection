# Active-Speaker-Detection
Dissertation project by Rory Williams and Supervised by Dr Yoshi Gotoh 

Aim of project is to create a lightweight vision-based active speaker detection appilcation

### Preparing the Dataset 
Dataset used is the [AVA-ActiveSpeaker Dataset]() by Roth et al  
  
To prepare in the dataset folder have a train and test folder each containing other folders named after the video ids that contain frames from that video as images with the title of 'videoID_timstamp'.jpg this can be done by executing the `/utils/data_prep.py` file by altering the video id input at the function call at the bottom of the file, you will also have to set teh boolean to show whether it is training or testing alternatively, this can be done using the ffmpeg command line package, frames are saved into a folder named after the video ID  

The videos are prepared at 10fps and 100-250 frames are used for each video  
Videos used:  
`train_ids = ['_mAfwH6i90E', 'B1MAUxpKaV8', '7nHkh4sP5Ks', '2PpxiG0WU18', '-5KQ66BBWC4', '5YPjcdLbs5g',
'20TAGRElvfE', 'Db19rWN5BGo', 'rFgb2ECMcrY', 'N0Dt9i9IUNg', '8aMv-ZGD4ic', 'Ekwy7wzLfjc', 
'0f39OWEqJ24']` 
 
`test_ids = ['4ZpjKfu6Cl8', '2qQs3Y9OJX0', 'HV0H6oc4Kvs', 'rJKeqfTlAeY', '1j20qq1JyX4', 'C25wkwAMB-w']`

### Running Code
To run the code:

Training:
`python train.py --{train, validate} --{SVM, MobileNet, ShuffleNet}`  
Validate will evaluate the model on the validation set after each training epoch (only to be used with MobileNet or ShuffleNet)  
There are additionally more arguements to change epochs, learningr rate and to display loss functions and cross-validation loss
  
Testing: 
`python train.py --test --{SVM, MobileNet, ShuffleNet} {--confMatrix} {--roc}`  

Additionally you can save results using the relevant arguement
 
#### Other Files
Juptyer Notebook `face_eval.ipynb` is used to perform evaluation and experimentation on the face detector