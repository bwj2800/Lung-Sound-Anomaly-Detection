import os
import tqdm
import numpy as np
import os
from utils import*
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm

import skimage.io as io
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure

# =============================================================================
# resizing, normalization and adaptive histogram equalization to images
# =============================================================================

class_names = ['wheeze', 'crackle','both','normal']

for class_name in class_names:
    source_dir='./data_4gr/original_images/'+class_name
    destination_dir='./data_4gr/original_images_preprocessed/'+class_name
    print("saving images to "+destination_dir)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        
    # list images list from source dir
    image_list=os.listdir(source_dir)# list of images
    # =============================================================================
    # Normalization and adaptive histogram equalization to each single image
    # =============================================================================
    # for img_name in image_list:
    for img_name in tqdm(image_list):
        img=io.imread(os.path.join(source_dir,img_name), as_gray=False)
        # print(img.shape)
        if len(img.shape) ==3:
            if img.shape[-1] == 4:
                # print('anh dau', img)
                img = img[:,:,:3]

            img_gray = rgb2gray(img)
        else:
            img_gray = img
        img_resized = resize(img_gray, (512, 512))#convert image size to 512*512
        img_rescaled=(img_resized-np.min(img_resized))/(np.max(img_resized)-np.min(img_resized)) # min-max normalization 
        img_enhanced=exposure.equalize_adapthist(img_rescaled) # adapt hist
        img_resized_8bit=img_as_ubyte(img_enhanced)
        io.imsave(os.path.join(destination_dir,img_name),img_resized_8bit)#save enhanced image to destination dir  
