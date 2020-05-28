# Mask-surveillance-Mask-detection-
Detects a person wearing a mask or not in real time and through image

   In this Covid-19 Pandemic situation, this Deep learning model can be used as a surveillance system to detect a person wearing a Mask or Not in real time and through static image. 

### Dataset:     
Create a Dataset with images of people wearing mask and without mask. Each having around 500 plus images. 

Here there are two main files,
* **training_mask.py** : This file takes the input Dataset and trains on the images using VGG16 architecture and the created Model is saved in the respective directory. 
* **detect_mask.py** : Used to detect face masks on the Static images using the saved model. 

### Training the Mask detector model:
* In the TrainingModel folder, there is **Training_mask.ipnb** and **training_mask.py** files. You can use either of the two files to train the model. 
* Pass the directory having Mask and WithoutMask folder to training.
* For training I have used VGG16 architecture and imagenet weights.
* Save the Model 

### Face Mask detection:
* Use **detect_mask.py** file for Face mask detection.
* Faces in the image are detected first using Opencv and prediction is made on top of this detected image. 
* ROI and detection label is created on the original image.
