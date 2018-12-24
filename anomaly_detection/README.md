## Anomaly detection using weakly labeled data 

* pseudo3dresnet.py : the main model
* data_preprocess.py : contains methods videos_array, frames, make_segments. 
    * videos_array is used for making an array of paths in a present in a directory.
    * frames returns the number of frames in a video
    * make_segments makes the video into 30 segments and again each of that segment into segments of 16 frames each and then reshapes to be compatable with input of model.    
