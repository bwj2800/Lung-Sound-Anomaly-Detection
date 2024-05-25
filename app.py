import gradio as gr
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from data_loader import DataLoader 
import librosa
import numpy as np
import pandas as pd
import os
import joblib
import wave


def make_prediction(output):
    if output==0:
        return "Normal"
    if output==1:
        return "Crackle"
    if output==2:
        return "Wheeze"
    else:
        return "Both(Crackle&Wheeze)"

def feature_extractor(audio, sample_rate, n_mels=64, f_min=50, f_max=2000, n_fft=2048, hop=512):
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
    
    # 만약 오디오 데이터가 스테레오(채널이 2개)인 경우, 모노(하나의 채널)로 변환
    if len(data.shape) == 2:
        # 스테레오 채널의 평균을 취하여 모노로 변환
        print("convert to mono")
        data_mono = np.mean(data, axis=1)
    else:
        data_mono = data
    
    feature_vector=feature_extractor(audio=data_mono, sample_rate=sr)
    model= load_model('checkpoint/model.h5')
    Y_Score=model.predict(feature_vector)
    print(":::Prediction made")   
    y_pred = np.argmax(Y_Score, axis=1)
    
    return make_prediction(y_pred)
    
demo = gr.Interface(fn=classify_respiratory_sound, inputs="audio", outputs="text")
 
demo.launch()
