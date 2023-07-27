# ShipChallenge
For the [Kaggel competition](https://www.kaggle.com/competitions/airbus-ship-detection/overview/) I have built Unet with the efficient net as the encoder using PyTorch.

The repo contains the following:
train.py - script for training - it accumulates training and validation loss and saves best models. If something interrupted training, you can start from the best-saved model.
data_prep.py - prepares datasets as pairs of image-mask
model_arc.py - contains model architecture and Early Stopping class
eval.py - contains helper function to visualize validation images predictions. It is saved to the folder 'report' together with a plot of learning curves.
inference.py - contains a function to predict any image.png. Some examples are saved to the folder 'predicted_test_vis'
requirements.txt - libraries used during the project

I have used Adam optimizer with the learning rate gradually decreasing over 100 iterations from 0.001 to 0.0001
![alt text](https://github.com/AnnPike/ShipChallenge/blob/main/LR_decay.png)
I have initialized the Early Stopping routine, where each iteration validation loss was checked. If it was larger than minimal for 5 epochs, training would be stopped. Due to time limits, I stopped training by myself with the best validation mean F2 score of 0.87 (the score is calculated as described on the competition page). So, the learning curves are:
![alt text](https://github.com/AnnPike/ShipChallenge/blob/main/model_lossdice_decay_62_final.png)


