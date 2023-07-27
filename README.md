# ShipChallenge
For the [Kaggel competition](https://www.kaggle.com/competitions/airbus-ship-detection/overview/) I have built Unet with the efficient net as the encoder and dice loss using PyTorch.<br />

The repo contains the following:<br />
train.py - script for training - it accumulates training and validation loss and saves best models. If something interrupted training, you can start from the best-saved model.<br />
data_prep.py - prepares datasets as pairs of image-mask <br />
model_arc.py - contains model architecture and Early Stopping class <br />
eval.py - contains helper functions to visualize the training process <br />
inference.py - contains a function to predict any image.png. Some examples are saved to the folder 'predicted_test_vis' <br />
requirements.txt - libraries used during the project <br />

I have used Adam optimizer with the learning rate gradually decreasing over 100 iterations from 0.001 to 0.0001<br />
![alt text](https://github.com/AnnPike/ShipChallenge/blob/main/LR_decay.png)<br />
I have initialized the Early Stopping routine, where each iteration validation loss was checked. If it was larger than minimal for 5 epochs, training would be stopped. Due to time limits, I stopped training by myself with the best validation mean F2 score of 0.87 (the score is calculated as described on the competition page). So, the learning curves are:<br />
![alt text](https://github.com/AnnPike/ShipChallenge/blob/main/model_lossdice_decay_final_report.png)
For a laptop with GPU Nvidia 3060 one epoch of training and vaditaion took ~40 mins.


