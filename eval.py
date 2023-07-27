import matplotlib.pyplot as plt
import numpy as np
from torch import sigmoid
import torchvision.transforms as T
import torch


def IoU_np(mask, pred_mask):
    """
  calculates IoU score for the ground truth mask and the predicted mask
  Args:
      mask (ndarray): ground truth mask
      pred_mask  (ndarray): predicted mask
  """
    mask_vect = mask.flatten().astype('bool')  # vector out of 2D array
    pred_mask_vect = pred_mask.flatten().astype('bool')  # vector out of 2D array
    inters = np.logical_and(mask_vect, pred_mask_vect).sum()  # amount of pixels in intersection
    union = np.logical_or(mask_vect, pred_mask_vect).sum()  # amount of pixels in union
    return (inters / union)


def iou(ten_mask, ten_pred):
    """
  calculates IoU score for the ground truth mask and the predicted mask for tensors
  Args:
      ten_mask (torch array): ground truth mask
      ten_pred  (torch array): predicted mask
  """
    l = ten_mask.shape[0]
    mask_vect = ten_mask.reshape(l, -1)
    pred_mask_vect = ten_pred.reshape(l, -1)
    inters = torch.logical_and(mask_vect, pred_mask_vect).sum(1)
    union = torch.logical_or(mask_vect, pred_mask_vect).sum(1)
    return inters / union


trans_back = T.Resize(768) #to get correct IoU score we need to resize masks to original shape


def plot_model_out_data(dataset, i_start, model, model_name):
    """
  plots and saves 4 images (with ships only) with corresponding true and predicted masks
  Args:
      dataset (nn Dataset): dataset to extract images from
      i_start  (int): number to start from
      model (torch model): model that predicts masks
      model_name (str): name of the model for saving
  """
    list_of_images = []
    list_of_masks = []
    list_of_masks_pred = []
    while len(list_of_images) < 4:
        image, mask = dataset[i_start]
        i_start = i_start + 1
        if mask.sum() == 0:
            continue

        model.eval()
        logits_mask = model(image.to('cuda').unsqueeze(0))
        pred_mask = sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5) * 1.0
        pred_mask = trans_back(pred_mask)
        pred_mask = np.transpose(pred_mask.squeeze(0).cpu().numpy(), (1, 2, 0)).astype(np.float32)

        mask = trans_back(mask)

        image = np.transpose(image.numpy(), (1, 2, 0)).astype(np.float32)
        mask = np.transpose(mask.numpy(), (1, 2, 0)).astype(np.float32).squeeze()

        list_of_images.append(image)
        list_of_masks.append(mask)
        list_of_masks_pred.append(pred_mask)

    plt.figure(figsize=(12, 12))
    for i in range(4):

        image = list_of_images[i]
        mask = list_of_masks[i]
        pred_mask = list_of_masks_pred[i]

        plt.subplot(4, 3, i * 3 + 1)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4, 3, i * 3 + 2)
        plt.imshow(pred_mask, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        IoUscore = IoU_np(mask, pred_mask)
        if pred_mask.sum() == 0:
            plt.title(f'PREDICTED no ships, FALSE NEGATIVE')
        else:
            plt.title(f'PREDICTED ship segmentation, IoU = {IoUscore:.2f}')

        plt.subplot(4, 3, i * 3 + 3)
        plt.imshow(mask, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if mask.sum() == 0:
            plt.title('no ships')
        else:
            plt.title('ship segmentation')

    plt.tight_layout()
    plt.savefig(f'reports/{model_name}_{i_start}')
    plt.close()


def plot_learning_curves(train_losses, valid_losses, valid_F2, epoch, model_name, loss, E=None):
    """
  plots and saves learning curves for a model training
  Args:
      train_losses (list): train losses over epochs
      valid_losses  (list): validation losses over epochs
      valid_F2 (list): F2 scores for validation dataset over epochs
      epoch (int): last epoch
      patience (int): early stopping patience
      model_name (str): model name to save the plot
      loss (str): loss may be 'dice' or 'dice_bse'
  """
    plt.figure(figsize=(20, 10))
    plt.plot(range(1, epoch + 2), train_losses, 'y', label='average train loss per batch')
    plt.plot(range(1, epoch + 2), valid_losses, 'b', label='average valid loss per batch')
    plt.text(epoch // 2, valid_losses[1], 'F2 valid score', weight='bold', fontsize=16)
    for i in range(1, epoch + 2, 5):
        plt.text(i, valid_losses[i - 1], str(np.round(valid_F2[i - 1].cpu().numpy(), 2)))
    if E:
        plt.axvline(x=E, color='r', linestyle="--", linewidth=1, label='early stopping')
        plt.text(E, valid_losses[E - 1]+0.01, str(np.round(valid_F2[E - 1].cpu().numpy(), 2)), weight='bold')
    else:
        plt.text(epoch + 1, valid_losses[epoch]+0.01, str(np.round(valid_F2[epoch].cpu().numpy(), 2)), weight='bold')
    plt.xlabel('# epochs')
    plt.ylabel(loss)
    plt.legend()
    plt.savefig(f'reports/{model_name}_final_report')
    plt.close()
