import albumentations as A #library for simultaneus image and mask augmentation
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import torch
from numpy import Inf

#define parameters
IMAGE_SIZE = 192
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'
LOSS = 'dice'

# augmentation function - only resize
def get_augs():
  return A.Resize(IMAGE_SIZE, IMAGE_SIZE)

# model architecture - Unet with specified ENCODER and DROPOUT ratio
class SegmentationModel(nn.Module):
  def __init__(self):
    super(SegmentationModel, self).__init__()
    self.arc = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None,
    )
  # we may chose 'dice loss' or combination of 'dice loss and bse'
  def forward(self, images, masks=None):
    logits = self.arc(images)
    if masks != None:
      loss1 = DiceLoss(mode='binary')(logits, masks)
      if LOSS == 'dice_bse':
          loss = nn.BCEWithLogitsLoss()
          loss2 = loss(logits, masks)
          return logits, loss1+loss2
      elif LOSS == 'dice':
          return logits, loss1
    return logits


# define ErlyStopping class, which saves model for the smallest validation loss
# if it doesn't improve over 'patience' epochs, it stops training
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience or if training loss doesn't improve after a given t_patience"""
    def __init__(self, patience=7, t_patience=5, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            t_patience  (int): How long to wait after train loss is not improving
                            Deafault: 5
            verbose (bool): If True, prints a message for each epoch
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