# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
import numpy
import torch
print(torch.cuda.is_available())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# data prep

import numpy as np
import pandas as pd
import torch
import model_arc as ma

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import cv2

DATA_DIR = 'data/'
CSV_FILE = DATA_DIR+'train_ship_segmentations_v2.csv'


def get_mask(encoded_str, image):
  '''returns a 2D array mask for the image'''
  h, w, _ = image.shape
  mask = np.zeros(h*w, dtype=int)

  if pd.isna(encoded_str):
    return mask.reshape(h, w)
  mask_encoded_arr = np.array(encoded_str.split(' '), dtype=int)
  start_pix = mask_encoded_arr[::2]
  count_pix = mask_encoded_arr[1::2]
  mask_idx = np.concatenate([np.arange(start_pix[i]-1, start_pix[i]+count_pix[i]-1) for i in range(len(count_pix))])

  mask[mask_idx] = 1
  return mask.reshape(w, h).T


df = pd.read_csv(CSV_FILE)
df_ship = df[~df.EncodedPixels.isna()]
df_no_ship = df[df.EncodedPixels.isna()]

df_ship_comb = df_ship.copy()
df_ship_comb.EncodedPixels = df_ship.EncodedPixels + ' '
df_ship_comb = df_ship_comb.pivot_table(values='EncodedPixels', index='ImageId', aggfunc='sum').reset_index()
df_ship_comb['EncodedPixels'] = df_ship_comb['EncodedPixels'].apply(lambda x: x.strip())

print(f'{len(df_ship_comb)} images contain ships')
print(f'{len(df_no_ship)} images do not contain any ships')

df_train_all = pd.concat((df_no_ship, df_ship_comb)).sample(frac=1, random_state=21)
df_train_all.head()


class SegmentationDataset(Dataset):
  def __init__(self, df, augmentations):
    self.df = df
    self.augmentations = augmentations
  def __len__(self):
    return len(self.df)
  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    image_path = row.ImageId
    mask_encoded = row.EncodedPixels

    image = cv2.imread(DATA_DIR+'train_v2/'+image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = get_mask(mask_encoded, image)
    mask = np.expand_dims(mask, axis=-1)
    if self.augmentations:
      data = self.augmentations(image=image, mask=mask)
      image = data['image']
      mask = data['mask']
    #(h, w, c) -> (c, h, w)

    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

    image = torch.Tensor(image)/255.0
    mask = torch.Tensor(mask)
    return image, mask

train_df, valid_df = train_test_split(df_train_all, test_size=0.2, random_state=21)

trainset = SegmentationDataset(train_df, ma.get_augs())
print(f"Size of Training Set : {len(trainset)}")

validset = SegmentationDataset(valid_df, ma.get_augs())
print(f"Size of Validation Set : {len(validset)}")


# eval
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
# model arc
import albumentations as A
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import torch
from numpy import Inf


IMAGE_SIZE = 192
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'
LOSS = 'dice'
DROPOUT = 0.5

def get_augs():
  return A.Resize(IMAGE_SIZE, IMAGE_SIZE)


class SegmentationModel(nn.Module):
  def __init__(self):
    super(SegmentationModel, self).__init__()

    aux_params = dict(
      dropout=DROPOUT,  # dropout ratio, default is None
      classes=0,  # define number of output labels
    )
    self.arc = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None
    )
  def forward(self, images, masks=None):
    logits = self.arc(images)
    if masks != None:
      loss1 = DiceLoss(mode='binary')(logits, masks)
      if LOSS == 'dice+bse':
          loss = nn.BCEWithLogitsLoss()
          loss2 = loss(logits, masks)
          return logits, loss1+loss2
      elif LOSS == 'dice':
          return logits, loss1
    return logits


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience or if training loss doesn't improve after a given t_patience"""
    def __init__(self, patience=5, t_patience=3, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            t_patience  (int): How long to wait after train loss is not improving
                            Deafault: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.t_patience = t_patience
        self.verbose = verbose
        self.counter = 0
        self.t_counter = 0
        self.best_score = None
        self.t_best_score = None
        self.early_stop = False
        self.val_loss_min = Inf
        self.path = path

    def __call__(self, score, t_score, model):

          if self.t_best_score is None:
            self.t_best_score = t_score
          elif t_score > self.t_best_score:
            self.t_best_score = t_score
            self.t_counter += 1
            if self.counter >= self.patience:
              self.early_stop = True
          else:
            self.t_best_score = t_score
            self.t_counter = 0

          if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
          elif score > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"Validation score didn't improve ({self.best_score:.4f} --> {score:.4f})., the patience is {self.counter}/{self.patience}")
            if self.counter >= self.patience:
              self.early_stop = True
          else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
# train
import torch
torch.cuda.empty_cache()
from tqdm import tqdm
from torch import sigmoid
from torch.utils.data import DataLoader
import model_arc as ma
import data_prep as dp
import eval


DEVICE = 'cuda'
PATIENCE = 2
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001

LOSS = ma.LOSS
DROPOUT = ma.DROPOUT

model = ma.SegmentationModel()
model.to(DEVICE)


def iou(ten_mask, ten_pred):
    l = ten_mask.shape[0]
    mask_vect = ten_mask.reshape(l, -1)
    pred_mask_vect = ten_pred.reshape(l, -1)
    inters = torch.logical_and(mask_vect, pred_mask_vect).sum(1)
    union = torch.logical_or(mask_vect, pred_mask_vect).sum(1)
    return inters/union


def train_fn(data_loader, model, optimizer):

  total_loss = 0.0

  for i, (images, masks) in enumerate(tqdm(data_loader)):
    if i == 10:
      break

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    model.train()
    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss/i


def eval_fn(data_loader, model):

  model.eval()
  total_loss = 0.0

  threshold_F2 = torch.arange(0.5, 1, 0.05)
  threshold_F2 = threshold_F2.reshape(10, 1).to(DEVICE)
  total_TP = torch.zeros(10).to(DEVICE)
  total_FN = torch.zeros(10).to(DEVICE)
  total_FP = torch.zeros(10).to(DEVICE)

  with torch.no_grad():
    for i, (images, masks) in enumerate(tqdm(data_loader)):
      if i == 10:
        break

      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

      logits, loss = model(images, masks)
      total_loss += loss.item()

      pred_masks = sigmoid(logits)
      pred_masks = (pred_masks > 0.5) * 1.0

      real_ships = masks.reshape(BATCH_SIZE, -1).sum(1) != 0
      pred_no_ships = pred_masks.reshape(BATCH_SIZE, -1).sum(1) == 0
      FN_batch = torch.logical_and(pred_no_ships, real_ships).sum()

      if real_ships.sum() != 0:
        masks_ships = masks[real_ships]
        pred_masks_ships = pred_masks[real_ships]
        iou_vect = iou(masks_ships, pred_masks_ships)
        table_th = iou_vect.repeat((10, 1)) >= threshold_F2
        TP_batch = table_th.sum(1)
        FP_batch = 1 - TP_batch
        total_TP += TP_batch
        total_FP += FP_batch

      total_FN += FN_batch.repeat(10)
  F2_vect = 5*total_TP/(5*total_TP+4*total_FN+total_FP)
  F2 = F2_vect.mean()
  return total_loss/i, F2


trainloader = DataLoader(dp.trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(dp.validset, batch_size=BATCH_SIZE)


early_stopping = ma.EarlyStopping(patience=PATIENCE, verbose=True)
model_name = 'new_model'
train_losses = []
valid_losses = []
valid_F2 = []
for epoch in range(EPOCHS):
  LR_decayed = LR * 1/(1 + LR*epoch/EPOCHS)
  print(f'\nTraining epoch {epoch+1} with learning rate: {LR_decayed}')
  optimizer = torch.optim.Adam(model.parameters(), lr=LR_decayed)
  train_loss = train_fn(trainloader, model, optimizer)
  valid_loss, F2 = eval_fn(validloader, model)
  train_losses.append(train_loss)
  valid_losses.append(valid_loss)
  valid_F2.append(F2)
  print(f'Epoch: {epoch + 1} Train loss: {train_loss} Valid loss: {valid_loss}, Valid F2: {F2}')
  early_stopping.path = F'best_models/{model_name}'
  early_stopping(valid_loss, train_loss, model)
  if early_stopping.early_stop:
    print("Early stopping")
    break

eval.plot_learning_curves(train_losses, valid_losses, valid_F2, epoch, PATIENCE, model_name, LOSS)
for i in range(0, len(dp.validset), 1000):
  eval.plot_model_out_data(dp.validset, i, model, model_name)

