import os
import numpy as np
import random
import tensorflow as tf
import pandas as pd
import cv2
import cmapy
from utils import*
import librosa
import librosa.display
from tqdm import tqdm
import torch
import scipy.io as sio
from scipy.stats import skew
from scipy.stats import kurtosis

import skimage.io as io

from IPython.display import Image

save_dir='./mat_new/'
from tensorflow.python.client import device_lib
print("============",device_lib.list_local_devices())
print("============",torch.cuda.is_available())
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


seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
tf.keras.utils.set_random_seed(seed_value)

sample_rate = 16000
desired_length = 5
# n_mels = 64
# nfft = 256
# hop = nfft//2
# f_max = 2000
n_mels = 64
nfft = 2048
hop = 512
f_min = 50
f_max = 2000

folds_file = './ICBHI_Dataset/patient_list_foldwise.txt'
# train_flag = train_flag
data_dir = './ICBHI_Dataset/audio_and_txt_files/'

def Extract_Annotation_Data(file_name, data_dir):
	tokens = file_name.split('_')
	recording_info = pd.DataFrame(data = [tokens], columns = ['Patient Number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
	recording_annotations = pd.read_csv(os.path.join(data_dir, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
	return recording_info, recording_annotations

# get annotations data and filenames
def get_annotations(data_dir):
	filenames = [s.split('.')[0] for s in os.listdir(data_dir) if '.txt' in s]
	i_list = []
	rec_annotations_dict = {}
	for s in filenames:
		i,a = Extract_Annotation_Data(s, data_dir)
		i_list.append(i)
		rec_annotations_dict[s] = a

	return filenames, rec_annotations_dict

def slice_data(start, end, raw_data, sample_rate):
	max_ind = len(raw_data) 
	start_ind = min(int(start * sample_rate), max_ind)
	end_ind = min(int(end * sample_rate), max_ind)
	return raw_data[start_ind: end_ind]

def get_label(crackle, wheeze):
	if crackle == 0 and wheeze == 0:
		return 0
	elif crackle == 1 and wheeze == 0:
		return 1
	elif crackle == 0 and wheeze == 1:
		return 2
	else:
		return 3

def get_sound_samples(recording_annotations, file_name, data_dir, sample_rate):
	sample_data = [file_name]
	# load file with specified sample rate (also converts to mono)
	data, rate = librosa.load(os.path.join(data_dir, file_name+'.wav'), sr=sample_rate)
	# print("Sample Rate", rate)
	
	for i in range(len(recording_annotations.index)):
		row = recording_annotations.loc[i]
		start = row['Start']
		end = row['End']
		crackles = row['Crackles']
		wheezes = row['Wheezes']
		audio_chunk = slice_data(start, end, data, rate)
		sample_data.append((audio_chunk, start,end, get_label(crackles, wheezes)))
	return sample_data

filenames, rec_annotations_dict = get_annotations(data_dir)
# print(rec_annotations_dict)

# =============================================================================
# Labeling data on a cyclical basis
# =============================================================================
filenames_with_labels = []
print("Extracting Individual Cycles")
cycle_list = []
classwise_cycle_list = [[], [], [],[]]
for idx, file_name in tqdm(enumerate(filenames)):
    data = get_sound_samples(rec_annotations_dict[file_name], file_name, data_dir, sample_rate)
    # print('data--------', data)
    # d[0]:audio_chunk, d[1]:start, d[2]:end, d[3]:get_label(crackles, wheezes)       data[1]:file name
    cycles_with_labels = [(d[0], d[3], file_name, cycle_idx, 0) for cycle_idx, d in enumerate(data[1:])] 
    # print('cycles_with_labels: ', cycles_with_labels)
    cycle_list.extend(cycles_with_labels)
    for cycle_idx, d in enumerate(cycles_with_labels):
        filenames_with_labels.append(file_name+'_'+str(d[3])+'_'+str(d[1]))
        classwise_cycle_list[d[1]].append(d)
print("len(cycle_list):",len(cycle_list))
print(len(classwise_cycle_list[0]),len(classwise_cycle_list[1]),len(classwise_cycle_list[2]),len(classwise_cycle_list[3]))


# =============================================================================
# Data augmentation
# =============================================================================
# augment normal
seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
tf.keras.utils.set_random_seed(seed_value)
scale = 1
aug_nos = scale*len(classwise_cycle_list[0]) - len(classwise_cycle_list[0])
for idx in range(aug_nos):
    # normal_i + normal_j
    i = random.randint(0, len(classwise_cycle_list[0])-1)
    j = random.randint(0, len(classwise_cycle_list[0])-1)
    normal_i = classwise_cycle_list[0][i]
    normal_j = classwise_cycle_list[0][j]
    new_sample = np.concatenate([normal_i[0], normal_j[0]])
    cycle_list.append((new_sample, 0, normal_i[2]+'-'+normal_j[2], idx, 1))
    filenames_with_labels.append(normal_i[2]+'-'+normal_j[2]+'_'+str(idx)+'_0')
    
# augment abnormal (crackle)
aug_nos = scale*len(classwise_cycle_list[0]) - len(classwise_cycle_list[1])
for idx in range(aug_nos):
    aug_prob = random.random()
    if aug_prob < 0.6:
        # crackle_i + crackle_j
        i = random.randint(0, len(classwise_cycle_list[1])-1)
        j = random.randint(0, len(classwise_cycle_list[1])-1)
        sample_i = classwise_cycle_list[1][i]
        sample_j = classwise_cycle_list[1][j]
    elif aug_prob >= 0.6 and aug_prob < 0.8:
        # crackle_i + normal_j
        i = random.randint(0, len(classwise_cycle_list[1])-1)
        j = random.randint(0, len(classwise_cycle_list[0])-1)
        sample_i = classwise_cycle_list[1][i]
        sample_j = classwise_cycle_list[0][j]
    else:
        # normal_i + crackle_j
        i = random.randint(0, len(classwise_cycle_list[0])-1)
        j = random.randint(0, len(classwise_cycle_list[1])-1)
        sample_i = classwise_cycle_list[0][i]
        sample_j = classwise_cycle_list[1][j]

    new_sample = np.concatenate([sample_i[0], sample_j[0]])
    cycle_list.append((new_sample, 1, sample_i[2]+'-'+sample_j[2], idx, 1))
    filenames_with_labels.append(sample_i[2]+'-'+sample_j[2]+'_'+str(idx)+'_1')

# augment abnormal (wheeze)
aug_nos = scale*len(classwise_cycle_list[0]) - len(classwise_cycle_list[2])
for idx in range(aug_nos):
    aug_prob = random.random()
    if aug_prob < 0.6:
        # wheeze_i + wheeze_j
        i = random.randint(0, len(classwise_cycle_list[2])-1)
        j = random.randint(0, len(classwise_cycle_list[2])-1)
        sample_i = classwise_cycle_list[2][i]
        sample_j = classwise_cycle_list[2][j]
    elif aug_prob >= 0.6 and aug_prob < 0.8:
        # wheeze_i + normal_j
        i = random.randint(0, len(classwise_cycle_list[2])-1)
        j = random.randint(0, len(classwise_cycle_list[0])-1)
        sample_i = classwise_cycle_list[2][i]
        sample_j = classwise_cycle_list[0][j]
    else:
        # normal_i + wheeze_j
        i = random.randint(0, len(classwise_cycle_list[0])-1)
        j = random.randint(0, len(classwise_cycle_list[2])-1)
        sample_i = classwise_cycle_list[0][i]
        sample_j = classwise_cycle_list[2][j]

    new_sample = np.concatenate([sample_i[0], sample_j[0]])
    cycle_list.append((new_sample, 2, sample_i[2]+'-'+sample_j[2], idx, 1))
    filenames_with_labels.append(sample_i[2]+'-'+sample_j[2]+'_'+str(idx)+'_2')


# augment abnormal (both)
aug_nos = scale*len(classwise_cycle_list[0]) - len(classwise_cycle_list[3])
for idx in range(aug_nos):
    aug_prob = random.random()
    if aug_prob < 0.5:
        # both_i + both_j
        i = random.randint(0, len(classwise_cycle_list[3])-1)
        j = random.randint(0, len(classwise_cycle_list[3])-1)
        sample_i = classwise_cycle_list[3][i]
        sample_j = classwise_cycle_list[3][j]
    elif aug_prob >= 0.5 and aug_prob < 0.7:
        # crackle_i + wheeze_j
        i = random.randint(0, len(classwise_cycle_list[1])-1)
        j = random.randint(0, len(classwise_cycle_list[2])-1)
        sample_i = classwise_cycle_list[1][i]
        sample_j = classwise_cycle_list[2][j]
    elif aug_prob >= 0.7 and aug_prob < 0.8:
        # wheeze_i + crackle_j
        i = random.randint(0, len(classwise_cycle_list[3])-1)
        j = random.randint(0, len(classwise_cycle_list[0])-1)
        sample_i = classwise_cycle_list[3][i]
        sample_j = classwise_cycle_list[0][j]
    elif aug_prob >= 0.8 and aug_prob < 0.9:
        # both_i + normal_j
        i = random.randint(0, len(classwise_cycle_list[3])-1)
        j = random.randint(0, len(classwise_cycle_list[0])-1)
        sample_i = classwise_cycle_list[3][i]
        sample_j = classwise_cycle_list[0][j]
    else:
        # normal_i + both_j
        i = random.randint(0, len(classwise_cycle_list[0])-1)
        j = random.randint(0, len(classwise_cycle_list[3])-1)
        sample_i = classwise_cycle_list[0][i]
        sample_j = classwise_cycle_list[3][j]

    new_sample = np.concatenate([sample_i[0], sample_j[0]])
    cycle_list.append((new_sample, 3, sample_i[2]+'-'+sample_j[2], idx, 1))
    filenames_with_labels.append(sample_i[2]+'-'+sample_j[2]+'_'+str(idx)+'_3')

print("len(cycle_list): ",len(cycle_list))


# =============================================================================
# Aligning to an 8-second duration
# =============================================================================
audio_data = [] # each sample is a tuple with id_0: audio_data, id_1: label, id_2: file_name, id_3: cycle id, id_4: aug id, id_5: split id
labels = []
desiredLength = 8
print('desiredLength*sample_rate: ', desiredLength*sample_rate)
output = []
for idx, sample in enumerate(cycle_list):
    # print(f'{idx}: {sample}')
    output_buffer_length = int(desiredLength*sample_rate)
    soundclip = sample[0].copy()
    # print('soundclip: ', soundclip)
    # d = soundclip[0:3]
    # b = np.concatenate((soundclip,d))
    # print('soundclip copy: ', b)
    n_samples = len(soundclip)
    # print('n_samples: ', n_samples)
    if n_samples < output_buffer_length: # shorter than 8sec
        t = output_buffer_length // n_samples
        # print('tttt', t)
        if output_buffer_length % n_samples == 0: # repeat sample
            repeat_sample = np.tile(soundclip, t)
            copy_repeat_sample = repeat_sample.copy()
            output.append((copy_repeat_sample, sample[1]))
        else: 
            d = output_buffer_length % n_samples
            # print('ddddd', d)
            d = soundclip[:d] # remainder
            # print('dddddddd: ', d)
            # print('soundclip*t:', len(np.tile(soundclip, t)), n_samples*t)
            repeat_sample = np.concatenate((np.tile(soundclip, t), d))
            copy_repeat_sample = repeat_sample.copy()
            # print('copy_repeat_sample:', len(copy_repeat_sample))
            output.append((copy_repeat_sample, sample[1]))
    else:  # longer than 8sec
        copy_repeat_sample = soundclip[:output_buffer_length]
        output.append((copy_repeat_sample, sample[1]))
print('----Len Output-----', len(output))        
# print('----Output-----', output[1][1])
audio_data.extend(output)
print('len audio data: ', len(audio_data))

# =============================================================================
# Saving as mel-spectrogram images
# =============================================================================
normal_feature_pool=np.empty([1,84])#feature pool [1,252]
crackle_feature_pool=np.empty([1,84])#feature pool [1,252]
wheeze_feature_pool=np.empty([1,84])#feature pool [1,252]
both_feature_pool=np.empty([1,84])#feature pool [1,252]
for index in tqdm(range(len(audio_data))): #len(audio_data)
    audio = audio_data[index][0]
    label = audio_data[index][1]    
    
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    mel_spec_db = librosa.power_to_db(S, ref=np.max)
    
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
    mfcc = librosa.feature.mfcc(S=mel_spec_db)
    mfcc_feature =  compute_14_features(mfcc) 
    
    feature_vector=np.concatenate((chroma_stft_feature, spectral_features, mfcc_feature), axis=0).reshape(1,84)
    if label==0:
        normal_feature_pool=np.concatenate((normal_feature_pool,feature_vector), axis=0)
    if label==1:
        crackle_feature_pool=np.concatenate((crackle_feature_pool,feature_vector), axis=0)
    if label==2:
        wheeze_feature_pool=np.concatenate((wheeze_feature_pool,feature_vector), axis=0)
    else:
        both_feature_pool=np.concatenate((both_feature_pool,feature_vector), axis=0)

normal_feature_pool=np.delete(normal_feature_pool, 0, 0)
crackle_feature_pool=np.delete(crackle_feature_pool, 0, 0)
wheeze_feature_pool=np.delete(wheeze_feature_pool, 0, 0)
both_feature_pool=np.delete(both_feature_pool, 0, 0)

#add label to the last column   
normal_feature_pool=np.concatenate((normal_feature_pool,0*np.ones(len(normal_feature_pool)).reshape(len(normal_feature_pool),1)), axis=1)
crackle_feature_pool=np.concatenate((crackle_feature_pool,1*np.ones(len(crackle_feature_pool)).reshape(len(crackle_feature_pool),1)), axis=1)
wheeze_feature_pool=np.concatenate((wheeze_feature_pool,2*np.ones(len(wheeze_feature_pool)).reshape(len(wheeze_feature_pool),1)), axis=1)
both_feature_pool=np.concatenate((both_feature_pool,3*np.ones(len(both_feature_pool)).reshape(len(both_feature_pool),1)), axis=1)

sio.savemat(save_dir+ 'normal_322.mat', {"normal": normal_feature_pool}) # save the created feature pool as a mat file 
sio.savemat(save_dir+ 'crackle_322.mat', {"crackle": crackle_feature_pool}) # save the created feature pool as a mat file 
sio.savemat(save_dir+ 'wheeze_322.mat', {"wheeze": wheeze_feature_pool}) # save the created feature pool as a mat file 
sio.savemat(save_dir+ 'both_322.mat', {"both": both_feature_pool}) # save the created feature pool as a mat file 
