import os
import librosa
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import cv2
import cmapy
import scipy
from scipy.stats import skew
from scipy.stats import kurtosis

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
f_max = 4000

folds_file = '../ICBHI_Dataset/patient_list_foldwise.txt'
# train_flag = train_flag
data_dir = '../ICBHI_Dataset/audio_and_txt_files/'

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


# =============================================================================
# compute_14_features
# =============================================================================
def compute_14_features(region):
    """ Compute 14 features """
    temp_array=region.reshape(-1)
    all_pixels=temp_array[temp_array!=0]
    
    # adding noise
    all_pixels += np.random.normal(0, 1e-8, all_pixels.shape) 
    
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


def create_mel_raw(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512, resz=1):
    # Mel-spectrogram
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Define filter parameters
    low_cutoff = 200
    high_cutoff = 1800
    band_low_cutoff = 200
    band_high_cutoff = 1800

    nyquist = 0.5 * sample_rate

    low = low_cutoff / nyquist
    b_low, a_low = scipy.signal.butter(4, low, btype='low')

    high = high_cutoff / nyquist
    b_high, a_high = scipy.signal.butter(4, high, btype='high')

    low_band = band_low_cutoff / nyquist
    high_band = band_high_cutoff / nyquist
    b_band, a_band = scipy.signal.butter(4, [low_band, high_band], btype='band')
    
    def apply_filter(S, b, a):
        return scipy.signal.lfilter(b, a, S, axis=0)
    # Apply filters
    S_low_filtered = np.apply_along_axis(lambda x: apply_filter(x, b_low, a_low), 0, S_db)
    S_high_filtered = np.apply_along_axis(lambda x: apply_filter(x, b_high, a_high), 0, S_db)
    S_band_filtered = np.apply_along_axis(lambda x: apply_filter(x, b_band, a_band), 0, S_db)
    
    filtered_signals = [S_low_filtered, S_high_filtered, S_band_filtered]
    
    # Extract Features
    spectral_images = []
    mfcc_images = []
    chroma_images = []
    
    for s in filtered_signals:
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(s))
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(s))
        spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(s))
        spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(s))
        spectral_flatness = librosa.feature.spectral_flatness(S=np.abs(s))
        
        spectral_feature = np.concatenate((
            compute_14_features(spectral_centroid), compute_14_features(spectral_bandwidth),
            compute_14_features(spectral_contrast), compute_14_features(spectral_rolloff),
            compute_14_features(spectral_flatness)
        ), axis=0)
        spectral_feature = (spectral_feature - spectral_feature.min()) / (spectral_feature.max() - spectral_feature.min())
        spectral_feature *= 255
        spectral_img = cv2.applyColorMap(spectral_feature.astype(np.uint8), cmapy.cmap('plasma'))
        spectral_img = cv2.flip(spectral_img, 0)
        
        if resz > 0:
            height, width, _ = spectral_img.shape
            spectral_img = cv2.resize(spectral_img, (width * resz, height * resz), interpolation=cv2.INTER_LINEAR)
        
        spectral_images.append(spectral_img)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(S=s, n_mfcc=12)
        energy = np.sum(librosa.feature.rms(y=current_window, frame_length=2048, hop_length=512)**2, axis=0)
        mfcc_with_energy = np.vstack((mfcc, energy))
        mfcc_feature = compute_14_features(mfcc_with_energy)
        mfcc_feature = (mfcc_feature - mfcc_feature.min()) / (mfcc_feature.max() - mfcc_feature.min())
        mfcc_feature *= 255
        mfcc_img = cv2.applyColorMap(mfcc_feature.astype(np.uint8), cmapy.cmap('viridis'))
        mfcc_img = cv2.flip(mfcc_img, 0)
        
        if resz > 0:
            height, width, _ = mfcc_img.shape
            mfcc_img = cv2.resize(mfcc_img, (width * resz, height * resz), interpolation=cv2.INTER_LINEAR)
        
        mfcc_images.append(mfcc_img)
        
        # Chroma features
        chroma_stft = librosa.feature.chroma_stft(S=np.abs(s))
        chroma_stft = (chroma_stft - chroma_stft.min()) / (chroma_stft.max() - chroma_stft.min())
        chroma_stft *= 255
        chroma_img = cv2.applyColorMap(chroma_stft.astype(np.uint8), cmapy.cmap('inferno'))
        chroma_img = cv2.flip(chroma_img, 0)
        
        if resz > 0:
            height, width, _ = chroma_img.shape
            chroma_img = cv2.resize(chroma_img, (width * resz, height * resz), interpolation=cv2.INTER_LINEAR)
        
        chroma_images.append(chroma_img)
        
    # Resize images to the same shape
    max_height = max(img.shape[0] for img in spectral_images + mfcc_images + chroma_images)
    max_width = max(img.shape[1] for img in spectral_images + mfcc_images + chroma_images)
    
    spectral_images = [cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_LINEAR) for img in spectral_images]
    mfcc_images = [cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_LINEAR) for img in mfcc_images]
    chroma_images = [cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_LINEAR) for img in chroma_images]
    
    # Combine all images into one 3x3 grid
    combined_img_top = np.hstack(spectral_images)
    combined_img_middle = np.hstack(mfcc_images)
    combined_img_bottom = np.hstack(chroma_images)
    combined_img = np.vstack([combined_img_top, combined_img_middle, combined_img_bottom])
    
    return combined_img
    
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
print('Creating images...')
mel_img = []
for index in tqdm(range(len(audio_data))): #len(audio_data)
    audio = audio_data[index][0]
    # label
    label = audio_data[index][1]    
    audio_image = cv2.cvtColor(
        create_mel_raw(audio, sample_rate, f_max= f_max, n_mels=n_mels, nfft=nfft, hop=hop, resz=1)
        , cv2.COLOR_BGR2RGB
    )
    mel_img_label = (audio_image, label)
    mel_img.append(mel_img_label)

destination_dir = '../data_4gr/mel_image_new/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
# Create the four folders for the labels
os.makedirs(os.path.join(destination_dir,'normal'), exist_ok=True)
os.makedirs(os.path.join(destination_dir,'crackle'), exist_ok=True)
os.makedirs(os.path.join(destination_dir,'wheeze'), exist_ok=True)
os.makedirs(os.path.join(destination_dir,'both'), exist_ok=True)

print('Saving images...')
for i in tqdm(range(len(mel_img))):
    input_data = mel_img[i][0]
    # print(input_data)
    labels = mel_img[i][1]
    # print(type(labels))
    
    if labels == 0: #1: abnormal, 0: normal
        cv2.imwrite(os.path.join(destination_dir,'normal', 'image_'+str(i)+'.jpg'), cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR))
    elif labels == 1:
        cv2.imwrite(os.path.join(destination_dir,'crackle', 'image_'+str(i)+'.jpg'), cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR))
    elif labels == 2:
        cv2.imwrite(os.path.join(destination_dir,'wheeze', 'image_'+str(i)+'.jpg'), cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(os.path.join(destination_dir,'both', 'image_'+str(i)+'.jpg'), cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR))
print('Done')