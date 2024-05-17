import numpy as np
import os
import random
from utils import*
import librosa
import librosa.display
from tqdm import tqdm

import scipy.io as sio
from scipy.stats import skew
from scipy.stats import kurtosis

import skimage.io as io

import tensorflow as tf

from IPython.display import Image

save_dir='./mat/'

if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

class_names = ['wheeze', 'crackle','both','normal']


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

    feature_pool=np.empty([1,84]) # feature pool [1,252]
    # for idx,img_name in enumerate(image_list):
    for idx, img_name in tqdm(enumerate(image_list), total=len(image_list)):
        
        # Extract Texture features
        # print(idx)
        img=io.imread(os.path.join(source_dir,img_name))
        img_rescaled=(img-np.min(img))/(np.max(img)-np.min(img))         


        
        mel_spec = librosa.feature.inverse.mel_to_stft(img_rescaled.astype(np.float32))
        # Convert to decibels
        mel_spec_db = librosa.amplitude_to_db(np.abs(mel_spec), ref=np.max)
        
        # Extract chroma feature
        chroma_stft = librosa.feature.chroma_stft(S=np.abs(mel_spec_db))
        chroma_stft_feature =  compute_14_features(chroma_stft) 
        
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
        
        feature_vector=np.concatenate((chroma_stft_feature, spectral_features, mfcc_feature), axis=0).reshape(1,84)
        feature_pool=np.concatenate((feature_pool,feature_vector), axis=0)

    feature_pool=np.delete(feature_pool, 0, 0)
    feature_pool=np.concatenate((feature_pool,label*np.ones(len(feature_pool)).reshape(len(feature_pool),1)), axis=1)#add label to the last column   
    sio.savemat(save_dir+ output_file_name + '_322.mat', {output_file_name: feature_pool}) # save the created feature pool as a mat file 
