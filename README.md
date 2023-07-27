# ShipChallenge
It is the solution to the Kaggel competition https://www.kaggle.com/competitions/airbus-ship-detection/overview
I have built Unet with the efficient net as the encoder using PyTorch.


The repo contains folowing:
I have used Adam optimizer wit the learning rate gradually decreasing over 100 iterations from 0.001 to 0.0001
![alt text](https://github.com/AnnPike/ShipChallenge/blob/main/LR_decay.png)
I have initialized the Early Stopping routine, where each iteration validation loss was checked. If it was larger than minimal for 5 epochs, training would be stopped. Due to time limits I stopped training by myself with the best validation mean F2 score of 0.87 (the score is calcualted as described in the competion page). So, the learning curves are:
![alt text](https://github.com/AnnPike/ShipChallenge/blob/main/model_lossdice_decay_62_final.png)


