import gradio as gr
import joblib
import librosa
import numpy as np
import os
import joblib
import cv2
import scipy
from scipy.stats import skew
from scipy.stats import kurtosis
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from PIL import Image

from torchvision import models, transforms, datasets

import cmapy

def align_length(sample, sample_rate=16000, desiredLength=10):
    print('desiredLength*sample_rate: ', desiredLength*sample_rate)
    output = []
    output_buffer_length = int(desiredLength*sample_rate)
    soundclip = sample.copy()
    n_samples = len(soundclip)
    
    if n_samples < output_buffer_length: # shorter than 8sec
        t = output_buffer_length // n_samples
        if output_buffer_length % n_samples == 0: # repeat sample
            repeat_sample = np.tile(soundclip, t)
            copy_repeat_sample = repeat_sample.copy()
        else:
            d = output_buffer_length % n_samples
            d = soundclip[:d] # remainder
            repeat_sample = np.concatenate((np.tile(soundclip, t), d))
            copy_repeat_sample = repeat_sample.copy()
    else:  # longer than 8sec
        copy_repeat_sample = soundclip[:output_buffer_length]
        
    return np.array(copy_repeat_sample)

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

def make_prediction(output):
    if output==0:
        return "Normal"
    if output==1:
        return "Crackle"
    if output==2:
        return "Wheeze"
    else:
        return "Both(Crackle&Wheeze)"

def get_mel_img(audio, sample_rate, n_mels=64, f_min=50, f_max=4000, nfft=2048, hop=512):    
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S_db = librosa.power_to_db(S, ref=np.max)

    S_img = (S_db-S_db.min()) / (S_db.max() - S_db.min())
    S_img *= 255
    img = cv2.applyColorMap(S_img.astype(np.uint8), cmapy.cmap('magma'))
    img = cv2.flip(img, 0)

    return img

def classify_respiratory_sound(audio):
    sr, data = audio
    
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.max(np.abs(data))
        
    sample_rate=16000
    data_resampled = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
    
    # librosa.load는 기본적으로 모노 채널 오디오를 반환합니다.
    # 만약 Gradio에서 받은 데이터가 스테레오라면, 모노로 변환해야 할 수 있습니다.
    if data_resampled.ndim > 1:
        data_mono = librosa.to_mono(data_resampled)
    else:
        data_mono = data_resampled
        
        
    print(data_mono.shape, data_mono.shape[0]/sample_rate)
    data_mono=align_length(data_mono, sample_rate=sample_rate)
    print(data_mono.shape, data_mono.shape[0]/sample_rate)
    mel_img = get_mel_img(audio=data_mono, sample_rate=sample_rate)
    
    
    label_map = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3}
    # ResNet 모델 사용
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(label_map))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load('checkpoint/best_resnet18_2.pth'))
    model.eval()
    
    # 이미지 변환 적용
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    print(type(mel_img), mel_img.shape)
    mel_img_transformed = transform(Image.fromarray(mel_img))
    print(type(mel_img_transformed), mel_img_transformed.shape)

    outputs = model(mel_img_transformed.unsqueeze(0).to(device))
    print("outputs",outputs)
    _, predicted = torch.max(outputs.data, 1)
    print("predicted",predicted)
    
    prediction = make_prediction(predicted)
    
    return prediction, mel_img

example_files = [
    "demo_audio/normal_104_1b1_Al_sc_Litt3200_segment_5.wav",
    "demo_audio/normal_104_1b1_Al_sc_Litt3200_segment_6.wav",
    
    "demo_audio/crackle_107_2b3_Ll_mc_AKGC417L_segment_4.wav",
    
    "demo_audio/wheeze_221_2b1_Pl_mc_LittC2SE_segment_3.wav",
    "demo_audio/wheeze_221_2b1_Pl_mc_LittC2SE_segment_5.wav",
    
    "demo_audio/both_107_2b3_Ar_mc_AKGC417L_segment_2.wav",
    "demo_audio/both_107_2b3_Ar_mc_AKGC417L_segment_4.wav",
    "demo_audio/breath2.wav",     
    "demo_audio/Crackle_1.wav",
    "demo_audio/Crackle_2.wav",     
    "demo_audio/Crackle_3.wav",     
    "demo_audio/Crackle_4.wav",     
    "demo_audio/Crackle_5.wav",     
    "demo_audio/Crackle_6.wav",       
    "demo_audio/Wheeze_1.wav",
    "demo_audio/Wheeze_2.wav",     
    "demo_audio/Wheeze_3.wav",          
    "demo_audio/Crackle_7_Lit.wav",
    "demo_audio/Crackle_8_Lit.wav",     
    "demo_audio/Crackle_9_Lit.wav",          
    "demo_audio/Wheeze_4_Lit.wav",
    "demo_audio/Wheeze_5_Lit.wav",     
    "demo_audio/Wheeze_6_Lit.wav",         
    "demo_audio/steth_wheeze.wav"  
]

demo = gr.Interface(
    fn=classify_respiratory_sound, 
    inputs=gr.Audio(),
    outputs=["text","image"],
    examples=example_files)

demo.launch()
