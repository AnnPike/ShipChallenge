import matplotlib.pyplot as plt
import numpy as np
from torch import sigmoid
from torch import load as torch_load
import torchvision.transforms as T
import model_arc as ma
import data_prep as dp


def IoU_np(mask, pred_mask):
  mask_vect = mask.flatten().astype('bool')
  pred_mask_vect = pred_mask.flatten().astype('bool')
  inters = np.logical_and(mask_vect, pred_mask_vect).sum()
  union = np.logical_or(mask_vect, pred_mask_vect).sum()
  return(inters/union)



trans_back = T.Resize((768, 768))


def plot_model_out_data(dataset, i_start, model, model_name):
  list_of_images = []
  list_of_masks = []
  list_of_masks_pred = []
  while len(list_of_images) < 4:
    print(len(list_of_images))
    image, mask = dataset[i_start]
    i_start = i_start + 1
    if mask.sum() == 0:
      continue

    model.eval()
    logits_mask = model(image.to('cuda').unsqueeze(0))
    pred_mask = sigmoid(logits_mask)
    pred_mask = (pred_mask > 0.5)*1.0
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

    plt.subplot(4, 3, i*3+1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(4, 3, i*3+2)
    plt.imshow(pred_mask, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    IoUscore = IoU_np(mask, pred_mask)
    if pred_mask.sum() == 0:
      plt.title(f'PREDICTED no ships, IoU = {IoUscore:.2f}')
    else:
      plt.title(f'PREDICTED ship segmentation, IoU = {IoUscore:.2f}')

    plt.subplot(4, 3, i*3+3)
    plt.imshow(mask, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if mask.sum() == 0:
      plt.title('no ships')
    else:
      plt.title('ship segmentation')

  plt.tight_layout()
  plt.savefig(f'reports/{model_name}_{i_start}')


def plot_learning_curves(train_losses, valid_losses, valid_F2, epoch, patience, model_name, loss):
  E = epoch+1-patience
  plt.plot(range(1, epoch + 2), train_losses, 'y', label='average train loss per batch')
  plt.plot(range(1, epoch + 2), valid_losses, 'b', label='average valid loss per batch')
  plt.text(E//2, valid_losses[1], 'F2 valid score')
  for i in range(1, epoch+2):
    plt.text(i, valid_losses[i-1], str(np.round(valid_F2[i-1].cpu().numpy(), 2)))
  plt.axvline(x=E, color='r', linestyle="--", linewidth=1, label='early stopping')
  plt.xlabel('# epochs')
  plt.ylabel(loss)
  plt.legend()
  plt.savefig(f'reports/{model_name}_final_report')
  plt.show()
