import os
import model_arc as ma
from torch import sigmoid
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import eval
import data_prep as dp
import pandas as pd

SAVE_TO = 'predicted_test_vis'

final_model = ma.SegmentationModel()
final_model.to('cuda')
IMAGE_SIZE = ma.IMAGE_SIZE
final_model.load_state_dict(torch.load('final_model'))

test_path = 'data/test_v2/'
transform = ma.get_augs()

# briefly observe many outputs for test set
test_df = pd.read_csv('data/sample_submission_v2.csv')
testset = dp.SegmentationDataset(test_df, ma.get_augs(), 'test_v2/')
for i in range(0, len(testset), 500):
    eval.plot_model_out_data(testset, i, final_model, 'test')


def predict(model, image):
    """
    Inference function. Returns a predicted mask for the image_p by the model
    Args:
        model (torch model): trained model
        image_p  (ndarray): image for which we calculate mask
    """
    image = transform(image=image)  # augment the image as during training
    image = np.transpose(image['image'], (2, 0, 1)).astype(np.float32)  # transpose for model input
    image = torch.Tensor(image).unsqueeze(0) / 255.0  # add batch dimention and standartize to have range of 0-1
    logits_mask = model(image.to('cuda'))  # get prediction and remove batch and chanel dimentions whicha are equal to 1
    pred_mask = sigmoid(logits_mask)  # sigmoid function
    pred_mask = (pred_mask > 0.5) * 1.0  # get binary mask
    pred_mask = np.transpose(pred_mask.squeeze(0).cpu().numpy(), (1, 2, 0)).astype(
        np.float32)  # get correct orientation numpy array
    return pred_mask


def plot_image_mask(image, mask, test_im):
    """
    plots and saves the image next to it's predicted mask
    Args:
        image (ndarray): image
        mask  (ndarray): predicted mask
        test_im (str): name of the image
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'test image {test_im}')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if mask.sum() == 0:
        plt.title(f'predicted no ships')
    else:
        plt.title(f'predicted segmentation')
    plt.tight_layout()
    plt.savefig(f'{SAVE_TO}/{test_im}')
    plt.close()


# save each 500's test image together with its prediction
for test_img in os.listdir(test_path)[::500]:
    image_p = cv2.imread(test_path + test_img)
    image_p = cv2.cvtColor(image_p, cv2.COLOR_BGR2RGB)
    pred_mask = predict(final_model, image_p)
    plot_image_mask(image_p, pred_mask, test_img)
