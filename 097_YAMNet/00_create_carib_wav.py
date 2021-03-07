
from scipy.io import wavfile
import tensorflow as tf
import numpy as np

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

wav_file_name = 'miaow_16k.wav'
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
duration = len(wav_data)/sample_rate

print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

# Normalization [-1.0, 1.0] = wav_data / 32767
waveform = wav_data / tf.int16.max
print(f'waveform: {waveform}')

# save .npy
waveform = np.asarray(waveform)
np.save('miaow_16k', waveform)
