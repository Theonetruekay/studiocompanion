from scipy.io import wavfile
import numpy as np
import torch
import librosa
from sklearn.preprocessing import StandardScaler

# Read WAV file
sample_rate, audio_data = wavfile.read("C:\\Users\\khwez\\Desktop\\Drake Best I Ever Had.wav")

# Convert stereo to mono
if len(audio_data.shape) == 2:
    audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)

# Ensure audio_data is float32 for librosa
audio_data = audio_data.astype(np.float32)

# Define segment length
segment_length = sample_rate  # 1 second

# Calculate total segments and pad if necessary
total_segments = len(audio_data) // segment_length
remainder = len(audio_data) % segment_length

# Pad audio data with zeros to make it perfectly divisible into segments
if remainder != 0:
    audio_data = np.pad(audio_data, (0, segment_length - remainder), 'constant', constant_values=0)

# Segment audio data
segments = np.array([audio_data[i:i + segment_length] for i in range(0, len(audio_data), segment_length)])

# Initialize feature arrays
mfccs = []
chroma = []
contrast = []

# Extract features for each segment
for segment in segments:
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
    chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sample_rate)

    mfccs.append(mfcc)
    chroma.append(chroma_stft)
    contrast.append(spectral_contrast)

# Convert list of numpy arrays to a single numpy array
mfccs_np = np.array(mfccs)
chroma_np = np.array(chroma)
contrast_np = np.array(contrast)

# Flatten the array for standardization
mfccs_reshaped = mfccs_np.reshape(-1, mfccs_np.shape[-1])

# Standardize
scaler = StandardScaler()
mfccs_scaled = scaler.fit_transform(mfccs_reshaped)

# Reshape back to original shape
mfccs_scaled = mfccs_scaled.reshape(mfccs_np.shape)

# Convert to PyTorch tensors
tensor_mfccs = torch.FloatTensor(mfccs_scaled)
tensor_chroma = torch.FloatTensor(chroma_np)
tensor_contrast = torch.FloatTensor(contrast_np)

# Save tensors
torch.save(tensor_mfccs, 'tensor_mfccs.pt')
