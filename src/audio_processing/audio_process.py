from pydub import AudioSegment

# Load an audio file
audio = AudioSegment.from_file("your_voice.wav")

# Resample to 16 kHz
audio = audio.set_frame_rate(16000)

# Split into segments
segments = [audio[start:end] for (start, end) in segment_ranges]
