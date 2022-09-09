import math
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from tensorflow import keras


def get_path(dataset:str ="train", img_type:str = "png", mask:bool = False):
    """dataset: train,validation,test; datatype:png,tiff """
    data_path = Path.cwd() / 'data'
    if mask:
        data_path = Path(data_path) / dataset / "masks" / "png"
        return data_path
    data_path = Path(data_path) / dataset / "images" / img_type
    return data_path

def tiling(image: np.ndarray, tile_size: int, stride:int = None):

    if stride is None: stride = tile_size
    img_height, img_width, channels = image.shape
    window = (tile_size,tile_size,channels)

    tiled_img = np.lib.stride_tricks.sliding_window_view(image, window)
    strided_img = tiled_img[::stride,::stride]
    strided_img = strided_img.reshape(-1,*window)
    return strided_img


class HubmapOrgan(keras.utils.Sequence):
    def __init__(self, batch_size, tile_size, stride=None, dataset:str ="train", img_type:str = "png", shuffle=False, augment=None):
        super().__init__()
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.stride = self.tile_size if stride is None else stride
        self.input_img_paths = list(get_path(dataset,img_type).iterdir())
        self.input_mask_paths = list(get_path(dataset,img_type, mask=True).iterdir())
        self.df = pd.read_csv(next(get_path(dataset,img_type).parent.parent.glob("*.csv")))
        self.padding = math.ceil(max(self.df["img_height"]/self.tile_size)) * tile_size
        self.shuffle = shuffle
        self.augment = augment
        self.tiles_number = ((self.padding - self.tile_size + self.stride ) // self.stride)**2
       
        
    def __len__(self):
        #return math.ceil(len(self.input_img_paths) / self.batch_size)
        return int(np.floor(len(self.input_img_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_mask_paths = self.input_mask_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size*self.tiles_number, self.tile_size, self.tile_size, 3), dtype='uint8')
        y = np.zeros((self.batch_size*self.tiles_number, self.tile_size, self.tile_size, 1), dtype='uint8')
        for j, (img_path,mask_path) in enumerate(zip(batch_img_paths,batch_mask_paths)):

            img = keras.utils.load_img(img_path)
            img_array = keras.utils.img_to_array(img, dtype='uint8')
            
            mask = keras.utils.load_img(mask_path, 
                                        color_mode="grayscale")
            mask_array = keras.utils.img_to_array(mask, dtype='uint8')

            pad = A.PadIfNeeded(min_height=self.padding, min_width=self.padding, p=1,always_apply=True)
            padded = pad(image=img_array, mask=mask_array)
            x[j*self.tiles_number:(j+1)*self.tiles_number] = tiling(padded['image'], self.tile_size, self.stride)
            y[j*self.tiles_number:(j+1)*self.tiles_number] = tiling(padded['mask'], self.tile_size, self.stride)
            
        if self.augment is None:
            return x.astype('float32')/255, y
        else:
            aug_x, aug_y = [], []
            for image, mask in zip(x, y):
                transformed = self.augment(image=image, mask=mask)
                aug_x.append(transformed['image'])
                aug_y.append(transformed['mask'])
            return np.array(aug_x), np.array(aug_y)
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.input_img_paths)
