import torch

torch.cuda.empty_cache()
from tqdm import tqdm
from torch import sigmoid
from torch.utils.data import DataLoader
import model_arc as ma
import data_prep as dp
import eval
import pickle
import numpy as np

DEVICE = 'cuda'
PATIENCE = 5
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001

LOSS = ma.LOSS

model = ma.SegmentationModel()
model.to(DEVICE)
IMAGE_SIZE = ma.IMAGE_SIZE
# my computer turned off by itself during training. I have saved best model, so I load the last best model - epoch 41
model.load_state_dict(torch.load('best_models/model_lossdice_decay_57'))


def train_fn(data_loader, model, optimizer):
    """
    runs one training epoch and returns mean loss per batch
    Args:
        data_loader (Dataloaer): training data
        model  (torch model): model that we update weights
        optimizer (torch optimizer): optimization algorithm
    """
    total_loss = 0.0

    for i, (images, masks) in enumerate(tqdm(data_loader)):
        # if i == 1:
        #     break

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        model.train()
        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (i + 1)


def eval_fn(data_loader, model):
    """
    runs one evaluation epoch and returns mean loss per batch and F2 score
    Args:
        data_loader (Dataloaer): validation data
        model  (torch model): model that we update weights
    """
    model.eval()
    total_loss = 0.0

    threshold_F2 = torch.arange(0.5, 1, 0.05)
    threshold_F2 = threshold_F2.reshape(10, 1).to(DEVICE)
    total_TP = torch.zeros(10).to(DEVICE)
    total_FN = torch.zeros(10).to(DEVICE)
    total_FP = torch.zeros(10).to(DEVICE)

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(data_loader)):
            # if i == 1:
            #     break

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits, loss = model(images, masks)
            total_loss += loss.item()

            pred_masks = sigmoid(logits)
            pred_masks = (pred_masks > 0.5) * 1.0

            real_ships = masks.reshape(-1, IMAGE_SIZE * IMAGE_SIZE).sum(1) != 0
            pred_no_ships = pred_masks.reshape(-1, IMAGE_SIZE * IMAGE_SIZE).sum(1) == 0
            FN_batch = torch.logical_and(pred_no_ships, real_ships).sum()

            if real_ships.sum() != 0:
                masks_ships = masks[real_ships]
                pred_masks_ships = pred_masks[real_ships]
                iou_vect = eval.iou(masks_ships, pred_masks_ships)
                table_th = iou_vect.repeat((10, 1)) >= threshold_F2
                TP_batch = table_th.sum(1)
                FP_batch = 1 - TP_batch
                total_TP += TP_batch
                total_FP += FP_batch

            total_FN += FN_batch.repeat(10)
    F2_vect = 5 * total_TP / (5 * total_TP + 4 * total_FN + total_FP)
    F2 = F2_vect.mean()
    return total_loss / (i + 1), F2


# define train loader and valid loader
trainloader = DataLoader(dp.trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(dp.validset, batch_size=BATCH_SIZE)
# define early stopping instance
early_stopping = ma.EarlyStopping(patience=PATIENCE, verbose=True)
# define model name to save models
model_name = f'model_loss{LOSS}_decay'
print(model_name)

# my computer turned off by itself during training. I have saved best model, so I load the previous losses
restore_dict = pickle.load(open(f'reports/model_loss{LOSS}_decay_pick', 'rb'))
train_losses = restore_dict['train_losses']
valid_losses = restore_dict['val_losses']
valid_F2 = restore_dict['F2_scores']
# last epoch ran was epoch last_epoch+1
last_epoch = len(train_losses)
print(last_epoch)
eval.plot_learning_curves(train_losses, valid_losses, valid_F2, last_epoch - 1, model_name, LOSS)

# I train with learning rate decaying slowly over time,
# since I start from epoch last_epoch+1, I need to calculate the decayed lr recursively
for i in range(last_epoch):
    # learning rate graduate decay each epoch. s.t. last epoch is about 10 times smaller than the first one
    LR = LR * 1 / (1 + LR * i)
    print(f'epoch {i + 1} ran with learning rate {LR}')
# load last best validation score from previous training session
val_min = np.array(valid_losses).min()
early_stopping.best_score = val_min
early_stopping.val_loss_min = val_min

# define dictionary to save the history of training
dict_to_save = {'val_losses': valid_losses, 'train_losses': train_losses, 'F2_scores': valid_F2}
for epoch in range(last_epoch, EPOCHS):
    # learning rate graduate decay each epoch. s.t. last epoch is about 10 times smaller than the first one
    LR = LR * 1 / (1 + LR * epoch)
    print(f'\nTraining epoch {epoch + 1} with learning rate: {LR}')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_loss = train_fn(trainloader, model, optimizer)
    # save some images to examine model performance
    for i in range(0, len(dp.validset), 1000):
        eval.plot_model_out_data(dp.validset, i, model, model_name)

    valid_loss, F2 = eval_fn(validloader, model)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_F2.append(F2)
    # each epoch save the losses and valid F2 score, so we can compare different models later
    pickle.dump(dict_to_save, open(f'reports/{model_name}_pick', 'wb'))

    print(f'Epoch: {epoch + 1} Train loss: {train_loss} Valid loss: {valid_loss}, Valid F2: {F2}')
    early_stopping.path = f'best_models/{model_name}_{epoch + 1}'
    early_stopping(valid_loss, train_loss, model)
    # if early stopping reached its patience, the training stops
    if early_stopping.early_stop:
        print("Early stopping")
        E = epoch + 1 - PATIENCE
        eval.plot_learning_curves(train_losses, valid_losses, valid_F2, epoch, model_name, LOSS, E)
        break
    else:
        eval.plot_learning_curves(train_losses, valid_losses, valid_F2, epoch, model_name, LOSS)
