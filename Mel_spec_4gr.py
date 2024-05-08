import os
import librosa
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import cv2
import cmapy

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

stetho_id=-1
folds_file = './ICBHI_Dataset/patient_list_foldwise.txt'
# train_flag = train_flag
data_dir = './ICBHI_Dataset/audio_and_txt_files/'
# file_name = './Dataset/audio_and_txt_files/'

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

	# recording_info = pd.concat(i_list, axis = 0)
	# recording_info.head()

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

def create_mel_raw(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512, resz=1):
	S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
	S = librosa.power_to_db(S, ref=np.max)
	S = (S-S.min()) / (S.max() - S.min())
	S *= 255
	img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
	height, width, _ = img.shape
	if resz > 0:
		img = cv2.resize(img, (width*resz, height*resz), interpolation=cv2.INTER_LINEAR)
	img = cv2.flip(img, 0)
	return img



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
    # print('--------', data)
    cycles_with_labels = [(d[0], d[3], file_name, cycle_idx, d[3]) for cycle_idx, d in enumerate(data[1:])] #label: d[3]
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
    cycle_list.append((new_sample, 0, normal_i[2]+'-'+normal_j[2], idx, 0))
    filenames_with_labels.append(normal_i[2]+'-'+normal_j[2]+'_'+str(idx)+'_0')
    
# augment abnormal
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
            output.append((copy_repeat_sample, sample[4]))
        else: 
            d = output_buffer_length % n_samples
            # print('ddddd', d)
            d = soundclip[:d] # remainder
            # print('dddddddd: ', d)
            # print('soundclip*t:', len(np.tile(soundclip, t)), n_samples*t)
            repeat_sample = np.concatenate((np.tile(soundclip, t), d))
            copy_repeat_sample = repeat_sample.copy()
            # print('copy_repeat_sample:', len(copy_repeat_sample))
            output.append((copy_repeat_sample, sample[4]))
    else:  # longer than 8sec
        copy_repeat_sample = soundclip[:output_buffer_length]
        output.append((copy_repeat_sample, sample[4]))
print('----Len Output-----', len(output))        
# print('----Output-----', output[1][1])
audio_data.extend(output)
print('len audio data: ', len(audio_data))


# =============================================================================
# Saving as mel-spectrogram images
# =============================================================================
mel_img = []
for index in range(len(audio_data)): #len(audio_data)
    audio = audio_data[index][0]
    # label
    label = audio_data[index][1]    
    audio_image = cv2.cvtColor(create_mel_raw(audio, sample_rate, f_max= f_max, 
            n_mels=n_mels, nfft=nfft, hop=hop, resz=3), cv2.COLOR_BGR2RGB)
    mel_img_label = (audio_image, label)
    mel_img.append(mel_img_label)
# for i in range(len(mel_img)):
#     print('mel_img: ', mel_img[i][1])
destination_dir = './data_4gr/original_images'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
# Create the four folders for the labels
os.makedirs(os.path.join(destination_dir,'normal'), exist_ok=True)
os.makedirs(os.path.join(destination_dir,'crackle'), exist_ok=True)
os.makedirs(os.path.join(destination_dir,'wheeze'), exist_ok=True)
os.makedirs(os.path.join(destination_dir,'both'), exist_ok=True)

for i in range(len(mel_img)):
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