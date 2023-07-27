import numpy as np
import pandas as pd
import torch
import model_arc as ma
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import cv2


DATA_DIR = 'data/'
CSV_FILE = DATA_DIR+'train_ship_segmentations_v2.csv'


#
def get_mask(encoded_str, image):
  """
  returns a 2D array mask for the image from string of encoded mask
  Args:
      encoded_str (str): string of encoded masks
      image  (array): image for which the mask is encoded
  """
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

#
class SegmentationDataset(Dataset):
  def __init__(self, df, augmentations, folder='train_v2/'):
    self.df = df
    self.augmentations = augmentations
    self.folder = folder
  def __len__(self):
    return len(self.df)
  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    image_path = row.ImageId
    mask_encoded = row.EncodedPixels

    image = cv2.imread(DATA_DIR+self.folder+image_path)
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

# read data frame with train data
df = pd.read_csv(CSV_FILE)
df_ship = df[~df.EncodedPixels.isna()] #samples with no ships
df_no_ship = df[df.EncodedPixels.isna()] #samples with ships
# some images contain multiple ships, the segmentation for each is in different row, let's combine it
df_ship_comb = df_ship.copy()
df_ship_comb.EncodedPixels = df_ship.EncodedPixels + ' '
df_ship_comb = df_ship_comb.pivot_table(values='EncodedPixels', index='ImageId', aggfunc='sum').reset_index()
df_ship_comb['EncodedPixels'] = df_ship_comb['EncodedPixels'].apply(lambda x: x.strip())

print(f'{len(df_ship_comb)} images contain ships')
print(f'{len(df_no_ship)} images do not contain any ships')
# combine all the images from the train_v2 into one data set and shuffle
df_train_all = pd.concat((df_no_ship, df_ship_comb)).sample(frac=1, random_state=21)
df_train_all.head()

# split the labeled dataset into train and validation set
train_df, valid_df = train_test_split(df_train_all, test_size=0.2, random_state=21)

#define torch trainset and validset
trainset = SegmentationDataset(train_df, ma.get_augs())
print(f"Size of Training Set : {len(trainset)}")

validset = SegmentationDataset(valid_df, ma.get_augs())
print(f"Size of Validation Set : {len(validset)}")