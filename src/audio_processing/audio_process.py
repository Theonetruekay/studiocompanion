from scipy.io import wavfile
import numpy as np
import torch

# Read WAV file
sample_rate, audio_data = wavfile.read("C:\\Users\\khwez\\Desktop\\Drake Best I Ever Had.wav")

# Assuming you want to work with 1-second segments
segment_length = sample_rate  # 1 second
total_segments = len(audio_data) // segment_length

# Segment audio data
segments = np.array([audio_data[i:i + segment_length] for i in range(0, len(audio_data), segment_length)])

# Convert segments to PyTorch tensors
tensor_segments = torch.FloatTensor(segments)

# Now tensor_segments is ready to be used in your PyTorch model
