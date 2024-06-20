import gradio as gr
from sklearn.model_selection import train_test_split

from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler
import joblib
import librosa
import numpy as np
import pandas as pd
import os
import joblib
import wave

import scipy
from scipy.stats import skew
from scipy.stats import kurtosis
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# def detect_respiratory_cycles(audio, sample_rate=16000):
#     # Use RMS energy to detect respiratory cycles
#     frame_length = int(0.1 * sample_rate)  # 100ms frame
#     hop_length = int(0.05 * sample_rate)  # 50ms hop
    
#     rms = librosa.feature.rms(audio, frame_length=frame_length, hop_length=hop_length)[0]
#     threshold = np.mean(rms) * 1.5  # You may need to adjust this threshold
    
#     cycles = []
#     current_cycle = []
#     is_inhale = False
    
#     for i, energy in enumerate(rms):
#         if energy > threshold and not is_inhale:
#             if current_cycle:
#                 cycles.append(current_cycle)
#             current_cycle = [i]
#             is_inhale = True
#         elif energy <= threshold and is_inhale:
#             current_cycle.append(i)
#             is_inhale = False
    
#     if current_cycle:
#         cycles.append(current_cycle)
    
#     cycle_audio_segments = []
#     for cycle in cycles:
#         start_frame = cycle[0] * hop_length
#         end_frame = cycle[-1] * hop_length + frame_length
#         cycle_audio_segments.append(audio[start_frame:end_frame])
    
#     return cycle_audio_segments

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
    # all_pixels += np.random.normal(0, 1e-8, all_pixels.shape) 
    all_pixels += 1e-8
    
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
    print(features.isnan())
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


def plot_feature(feature, title):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(feature, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.close()
    return plt

def feature_extractor(audio, sample_rate, n_mels=64, f_min=50, f_max=4000, nfft=2048, hop=512):    
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Apply Mel-Spectrogram Filter
    # Define filter parameters
    low_cutoff = 200  # Lowpass filter cutoff frequency (in Hz)
    high_cutoff = 1800  # Highpass filter cutoff frequency (in Hz)
    band_low_cutoff = 200  # Bandpass filter lower cutoff frequency (in Hz)
    band_high_cutoff = 1800  # Bandpass filter upper cutoff frequency (in Hz)

    # Calculate Nyquist frequency
    nyquist = 0.5 * sample_rate

    # Design filters
    low = low_cutoff / nyquist
    b_low, a_low = scipy.signal.butter(4, low, btype='low')

    high = high_cutoff / nyquist
    b_high, a_high = scipy.signal.butter(4, high, btype='high')

    low_band = band_low_cutoff / nyquist
    high_band = band_high_cutoff / nyquist
    b_band, a_band = scipy.signal.butter(4, [low_band, high_band], btype='band')

    # Apply filters to the Mel spectrogram
    def apply_filter(S, b, a):
        return scipy.signal.lfilter(b, a, S, axis=0)

    # Apply filters to each frequency bin in the Mel spectrogram
    S_low_filtered = np.apply_along_axis(lambda x: apply_filter(x, b_low, a_low), 0, S_db)
    S_high_filtered = np.apply_along_axis(lambda x: apply_filter(x, b_high, a_high), 0, S_db)
    S_band_filtered = np.apply_along_axis(lambda x: apply_filter(x, b_band, a_band), 0, S_db)
    
    S_filtered = np.stack((S_low_filtered, S_high_filtered, S_band_filtered), axis=-1)

    # Extract Features
    spectral_features = []
    mfcc_features = []
    chroma_features = []
    poly_features = []
    for s in [S_low_filtered, S_high_filtered, S_band_filtered]:
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(s))
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(s))
        spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(s))
        spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(s))
        spectral_flatness = librosa.feature.spectral_flatness(S=np.abs(s))
        
        spectral_feature=np.concatenate((compute_14_features(spectral_centroid),        compute_14_features(spectral_bandwidth), compute_14_features(spectral_contrast),compute_14_features(spectral_rolloff), compute_14_features(spectral_flatness)), axis=0)
        spectral_features.append(spectral_feature)
        
        # Extract MFCC feature
        mfcc = librosa.feature.mfcc(S=s,n_mfcc=12)
        energy = np.sum(librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)**2, axis=0)
        mfcc_with_energy = np.vstack((mfcc, energy))
        mfcc_feature =  compute_14_features(mfcc_with_energy)
        mfcc_features.append(mfcc_feature)
        
        # Extract chroma feature
        chroma_stft = librosa.feature.chroma_stft(S=np.abs(s)) 
        chroma_cqt = librosa.feature.chroma_cqt(C=np.abs(s))
        chroma_cens = librosa.feature.chroma_cens(C=np.abs(s))
        chroma_vqt = librosa.feature.chroma_vqt(V=np.abs(s), intervals='ji5')
        chroma_feature=np.concatenate((compute_14_features(chroma_stft), compute_14_features(chroma_cqt),compute_14_features(chroma_cens),compute_14_features(chroma_vqt)), axis=0)
        # chroma_stft_feature =  compute_14_features(chroma_stft)
        chroma_features.append(chroma_feature)
        
        p0 = librosa.feature.poly_features(S=np.abs(s), order=0)
        p0_feature =  compute_14_features(p0) 
        poly_features.append(p0_feature)
    
    total_spectral_features = np.concatenate((spectral_features[0],spectral_features[1],spectral_features[2]),axis=0)
    total_chroma_features = np.concatenate((chroma_features[0],chroma_features[1],chroma_features[2]), axis=0)
    total_mfcc_features = np.concatenate((mfcc_features[0],mfcc_features[1],mfcc_features[2]), axis=0)
    total_poly_features = np.concatenate((poly_features[0],poly_features[1],poly_features[2]), axis=0)
     
    num_features=462
    feature_vector=np.concatenate((total_chroma_features, total_spectral_features, total_mfcc_features, total_poly_features), axis=0).reshape(1,num_features)
    
    return feature_vector

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
    
    scaler_path ="./checkpoint/scaler252.pkl"
    transformer_path="./checkpoint/transformer252.pkl"
    min_max_scaler = joblib.load(scaler_path)
    transformer = joblib.load(transformer_path)
    model= load_model('checkpoint/model252.h5')
        
    print(data_mono.shape, data_mono.shape[0]/sample_rate)
    data_mono=align_length(data_mono, sample_rate=sample_rate)
    print(data_mono.shape, data_mono.shape[0]/sample_rate)
    feature_vector=feature_extractor(audio=data_mono, sample_rate=sample_rate)
    
    print(min(feature_vector[0]),max(feature_vector[0]))
    
    X = min_max_scaler.transform(feature_vector)
    X = transformer.transform(X)
    print("X.shape",X.shape)
    
    np.set_printoptions(precision=5, suppress=True)
    print(X[0,:15])
    
    Y_Score=model.predict(X)
    print(":::Prediction made")   
    print(Y_Score)
    y_pred = np.argmax(Y_Score, axis=1)
    # results.append(make_prediction(y_pred))
    
    return make_prediction(y_pred)
    
example_files = [
    "ICBHI_Dataset/audio_and_txt_files/104_1b1_Al_sc_Litt3200.wav",
    "ICBHI_Dataset/audio_and_txt_files/107_2b3_Ll_mc_AKGC417L.wav",
    "ICBHI_Dataset/audio_and_txt_files/221_2b1_Pl_mc_LittC2SE.wav",
    "ICBHI_Dataset/audio_and_txt_files/107_2b3_Ar_mc_AKGC417L.wav"    
]

demo = gr.Interface(
    fn=classify_respiratory_sound, 
    inputs=gr.Audio(),
    outputs=["text", "plot", "plot", "plot", "plot"],
    examples=example_files)
 
demo.launch()
