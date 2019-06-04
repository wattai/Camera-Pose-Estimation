# Task1-LabRetreat

# Environment 
python==3.6  
numpy==1.16.4  
opencv==4.1.0  
matplotlib==2.2.2  

# How to run
0. Calibrate camera using camera_calibration.py
1. Put 9*6 checkerboard in the range of webcam
2. Run capture_video.py to take video-file which focus on the checkerboard as the training dataset
3. Take video around the checkerboard
4. Push key "q" on the opencv-popup-window
5. Run pose_estimator.py

# Result
You can see the error of the estimated camera-poses between the pose based on checkerboard adn the pose based on natural-feature.
![an example of result](https://github.com/wattai/Task1-LabRetreat/blob/master/task1/result.PNG "result")
