import gradio as gr
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import pandas as pd
import os
import joblib
import wave
from scipy.stats import skew
from scipy.stats import kurtosis
from tensorflow.keras.models import load_model

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



def make_prediction(output):
    if output==0:
        return "Normal"
    if output==1:
        return "Crackle"
    if output==2:
        return "Wheeze"
    else:
        return "Both(Crackle&Wheeze)"

def feature_extractor(audio, sample_rate, n_mels=64, f_min=50, f_max=2000, nfft=2048, hop=512):
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
    
    return feature_vector
def classify_respiratory_sound(audio):
    sr, data = audio
    
    # 데이터가 정수형인 경우 부동 소수점으로 정규화
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    
    # 데이터를 부동 소수점 형식으로 변환 (이미 부동 소수점인 경우 무시됨)
    data_float32 = data.astype(np.float32)
    data_resampled = librosa.resample(data_float32, orig_sr=sr, target_sr=16000)
    
    # librosa.load는 기본적으로 모노 채널 오디오를 반환합니다.
    # 만약 Gradio에서 받은 데이터가 스테레오라면, 모노로 변환해야 할 수 있습니다.
    if data_resampled.ndim > 1:
        data_mono = librosa.to_mono(data_resampled)
    else:
        data_mono = data_resampled
    
    feature_vector=feature_extractor(audio=data_mono, sample_rate=sr)
    model= load_model('checkpoint/model_2.h5')
    Y_Score=model.predict(feature_vector)
    print(":::Prediction made")   
    y_pred = np.argmax(Y_Score, axis=1)
    
    return make_prediction(y_pred)
    
example_files = [
    "ICBHI_Dataset/audio_and_txt_files/104_1b1_Al_sc_Litt3200.wav"
    "ICBHI_Dataset/audio_and_txt_files/107_2b3_Ll_mc_AKGC417L.wav",
    "ICBHI_Dataset/audio_and_txt_files/221_2b1_Pl_mc_LittC2SE.wav",
    "ICBHI_Dataset/audio_and_txt_files/107_2b3_Ar_mc_AKGC417L.wav"    
]

demo = gr.Interface(
    fn=classify_respiratory_sound, 
    inputs=gr.Audio(),
    outputs="text",
    examples=example_files)
 
demo.launch()
