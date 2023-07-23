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
model_name = f'drop{DROPOUT}_loss{LOSS}'
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

