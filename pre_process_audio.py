import librosa
import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
from PIL import Image
import io

def generate_mel_spectrogram(audio_path, output_path="mel_spectrogram.png"):
    # Audio playback code...
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    
    print("Audio is playing... Press Enter to continue to spectrogram generation")
    input()  # Wait for user input
    
    # GTZAN-compatible parameters
    sampling_rate = 22050
    n_fft = 2048
    hop_length = 512
    n_mels = 128

    y, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
    
    # Generate mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    
    # Convert to decibels - this is the key change
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Normalize to 0-1 range for better visualization
    S_dB_normalized = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    
    # Key changes: use 'plasma' colormap and proper aspect ratio
    plt.imshow(S_dB_normalized, 
               aspect='auto', 
               origin='lower',
               cmap='viridis_r',  # Changed to plasma for purple colors
               interpolation='bilinear')
    
    plt.tight_layout(pad=0.33)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=1, dpi=150)
    plt.show()
    
    print(f"Saved mel spectrogram as {output_path}")
    
    
    
    return output_path