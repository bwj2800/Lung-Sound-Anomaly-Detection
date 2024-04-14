import numpy as np
import seaborn as sn
import pandas as pd
import os
import pywt
import random
from utils import*
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm

import scipy.io as sio
from scipy.stats import skew
from scipy.stats import kurtosis

from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import skimage.io as io
from skimage.feature import graycomatrix
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import layers as kl

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from IPython.display import Image
import itertools


seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
# tf.keras.utils.set_random_seed(seed_value)
# tf.random.set_seed()


def im2double(img):
    """ convert image to double format """
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype('float') - min_val) / (max_val - min_val)
    return out

# =============================================================================
# compute_14_features
# =============================================================================
def compute_14_features(region):
    """ Compute 14 features """
    temp_array=region.reshape(-1)
    all_pixels=temp_array[temp_array!=0]
#    Area
    Area = np.sum(all_pixels)
#    mean
    density = np.mean(all_pixels)
#   Std
    std_Density = np.std(all_pixels)
#   skewness
    Skewness = skew(all_pixels)
#   kurtosis
    Kurtosis = kurtosis(all_pixels)
#   Energy
    ENERGY =np.sum(np.square(all_pixels))
#   Entropy
    value,counts = np.unique(all_pixels, return_counts=True)
    p = counts / np.sum(counts)
    p =  p[p!=0]
    ENTROPY =-np.sum( p*np.log2(p))
#   Maximum
    MAX = np.max(all_pixels)
#   Mean Absolute Deviation
    sum_deviation= np.sum(np.abs(all_pixels-np.mean(all_pixels)))
    mean_absolute_deviation = sum_deviation/len(all_pixels)
#   Median
    MEDIAN = np.median(all_pixels)
#   Minimum
    MIN = np.min(all_pixels)
#   Range
    RANGE = np.max(all_pixels)-np.min(all_pixels)
#   Root Mean Square
    RMS = np.sqrt(np.mean(np.square(all_pixels))) 
#    Uniformity
    UNIFORMITY = np.sum(np.square(p))

    features = np.array([Area, density, std_Density,
        Skewness, Kurtosis,ENERGY, ENTROPY,
        MAX, mean_absolute_deviation, MEDIAN, MIN, RANGE, RMS, UNIFORMITY])
    return features

# =============================================================================
# GLDM
# =============================================================================
def GLDM(img, distance):
    """ GLDM in four directions """
    pro1=np.zeros(img.shape,dtype=np.float32)
    pro2=np.zeros(img.shape,dtype=np.float32)
    pro3=np.zeros(img.shape,dtype=np.float32)
    pro4=np.zeros(img.shape,dtype=np.float32)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            if((j+distance)<img.shape[1]):
                pro1[i,j]=np.abs(img[i,j]-img[i,(j+distance)])
            if((i-distance)>0)&((j+distance)<img.shape[1]):
                pro2[i,j]=np.abs(img[i,j]-img[(i-distance),(j+distance)])
            if((i+distance)<img.shape[0]):
                pro3[i,j]=np.abs(img[i,j]-img[(i+distance),j])
            if((i-distance)>0)&((j-distance)>0):
                pro4[i,j]=np.abs(img[i,j]-img[(i-distance),(j-distance)])

    n=256
    cnt, bin_edges=np.histogram(pro1[pro1!=0], bins=np.arange(n)/(n-1), density=False)
    Out1 = cnt.cumsum()
    cnt, bin_edges=np.histogram(pro2[pro2!=0], bins=np.arange(n)/(n-1), density=False)
    Out2 = cnt.cumsum()
    cnt, bin_edges=np.histogram(pro3[pro3!=0], bins=np.arange(n)/(n-1), density=False)
    Out3 = cnt.cumsum()
    cnt, bin_edges=np.histogram(pro4[pro4!=0], bins=np.arange(n)/(n-1), density=False)
    Out4 = cnt.cumsum()
    return Out1,Out2,Out3,Out4


# =============================================================================
# resizing, normalization and adaptive histogram equalization to images
# =============================================================================

class_names = ['wheeze', 'crackle','both','normal']
# for class_name in class_names:
#     source_dir='./data_4gr/original_images/'+class_name
#     destination_dir='./data_4gr/original_images_preprocessed/'+class_name
#     print("saving images to "+destination_dir)

#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)
        
#     # list images list from source dir
#     image_list=os.listdir(source_dir)# list of images
#     # =============================================================================
#     # Normalization and adaptive histogram equalization to each single image
#     # =============================================================================
#     # for img_name in image_list:
#     for img_name in tqdm(image_list):
#         img=io.imread(os.path.join(source_dir,img_name), as_gray=False)
#         # print(img.shape)
#         if len(img.shape) ==3:
#             if img.shape[-1] == 4:
#                 # print('anh dau', img)
#                 img = img[:,:,:3]

#             img_gray = rgb2gray(img)
#         else:
#             img_gray = img
#         img_resized = resize(img_gray, (512, 512))#convert image size to 512*512
#         img_rescaled=(img_resized-np.min(img_resized))/(np.max(img_resized)-np.min(img_resized)) # min-max normalization 
#         img_enhanced=exposure.equalize_adapthist(img_rescaled) # adapt hist
#         img_resized_8bit=img_as_ubyte(img_enhanced)
#         io.imsave(os.path.join(destination_dir,img_name),img_resized_8bit)#save enhanced image to destination dir  


# =============================================================================
# feature extraction to create feature pool
# =============================================================================
for class_name in class_names:
    source_dir='./data_4gr/original_images_preprocessed/'+class_name
    output_file_name=class_name

    # set labels
    if output_file_name=='normal':
        label=0
    elif output_file_name=='crackle':
        label=1
    elif output_file_name=='wheeze':
        label=2    
    else:
        label=3

    # start
    image_list=os.listdir(source_dir) # list of images

    feature_pool=np.empty([1,322]) # feature pool [1,252]
    # for idx,img_name in enumerate(image_list):
    for idx, img_name in tqdm(enumerate(image_list), total=len(image_list)):
        # Extract Texture features
        # print(idx)
        img=io.imread(os.path.join(source_dir,img_name))
        img_rescaled=(img-np.min(img))/(np.max(img)-np.min(img)) 
        # print(img.shape)
        texture_features=compute_14_features(img_rescaled) # texture features
        
        # Extract FFT features
        fft_map=np.fft.fft2(img_rescaled)
        fft_map = np.fft.fftshift(fft_map)
        fft_map = np.abs(fft_map)
        YC=int(np.floor(fft_map.shape[1]/2)+1)
        fft_map=fft_map[:,YC:int(np.floor(3*YC/2))]
        # print('fft_map: ', fft_map.shape)
        fft_features=compute_14_features(fft_map) #FFT features
        
        # Extract Wavelet features
        wavelet_coeffs = pywt.dwt2(img_rescaled,'sym4')
        cA1, (cH1, cV1, cD1) = wavelet_coeffs
        # print('cA1: ', cA1.shape)
        # print('cH1: ', cH1.shape)
        wavelet_coeffs = pywt.dwt2(cA1,'sym4')
        cA2, (cH2, cV2, cD2) = wavelet_coeffs #wavelet features
        wavelet_features=np.concatenate((compute_14_features(cA1), compute_14_features(cH1),compute_14_features(cV1),compute_14_features(cD1)
        ,compute_14_features(cA2), compute_14_features(cH2),compute_14_features(cV2),compute_14_features(cD2)), axis=0)
        
        # Extract GLDM features
        gLDM1,gLDM2,gLDM3,gLDM4=GLDM(img_rescaled,10) # GLDM in four directions
        # print('gLDM1: ', gLDM1.shape)
        # hka = s
        gldm_features=np.concatenate((compute_14_features(gLDM1), compute_14_features(gLDM2),
                                    compute_14_features(gLDM3),compute_14_features(gLDM4)), axis=0)
        
        # Extract GLCM features
        glcms =graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])#GLCM in four directions
        glcm_features=np.concatenate((compute_14_features(im2double(glcms[:, :, 0, 0])), 
                                    compute_14_features(im2double(glcms[:, :, 0, 1])),
                                    compute_14_features(im2double(im2double(glcms[:, :, 0, 2]))),
                                    compute_14_features(glcms[:, :, 0, 3])), axis=0)
        

        mel_spec = librosa.feature.inverse.mel_to_stft(img_rescaled.astype(np.float32))
        # Convert to decibels
        mel_spec_db = librosa.amplitude_to_db(np.abs(mel_spec), ref=np.max)

        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(mel_spec_db))
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(mel_spec_db))
        spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(mel_spec_db))
        spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(mel_spec_db))
        
        spectral_features=np.concatenate((compute_14_features(spectral_centroid), compute_14_features(spectral_bandwidth),
                                    compute_14_features(spectral_contrast),compute_14_features(spectral_rolloff)), axis=0)
        # Extract MFCC feature
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec_db))
        mfcc_feature =  compute_14_features(mfcc) 
        
        feature_vector=np.concatenate((texture_features,fft_features,wavelet_features,gldm_features,glcm_features, spectral_features, mfcc_feature), axis=0).reshape(1,322)#merge to create a feature vector of 252
        feature_pool=np.concatenate((feature_pool,feature_vector), axis=0)

    feature_pool=np.delete(feature_pool, 0, 0)
    feature_pool=np.concatenate((feature_pool,label*np.ones(len(feature_pool)).reshape(len(feature_pool),1)), axis=1)#add label to the last column   
    sio.savemat(output_file_name + '_322.mat', {output_file_name: feature_pool}) # save the created feature pool as a mat file 
