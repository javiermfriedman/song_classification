import librosa
import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
from PIL import Image
import io


def generate_mel_spectrogram(audio_path, output_path="mel_spectrogram_clean.png"):
    
    # Audio playback
    # pygame.mixer.init()
    # pygame.mixer.music.load(audio_path)
    # pygame.mixer.music.play()
    
    # print("Audio is playing... Press Enter to continue to spectrogram generation")
    # input()  
    
    # Load audio - keeping all the detail
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Generate mel spectrogram with FULL detail
    mel_spect = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=2048,      # Keep full frequency resolution
        hop_length=512,   # Keep full time resolution  
        n_mels=128       # Keep all mel bands
    )
    
    # Convert to dB
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    print(f"Full detail mel spectrogram shape: {mel_spect_db.shape}")
    
    # Create visualization with SMALLER image output
    plt.figure(figsize=(6, 3))  # Smaller figure size
    plt.axis('off')
    
    plt.imshow(
        mel_spect_db, 
        aspect='auto', 
        origin='lower',
        cmap='viridis',
        interpolation='bilinear'
    )
    
    plt.tight_layout(pad=0)
    
    # Save with lower DPI to create smaller file
    plt.savefig(output_path, 
                bbox_inches='tight', 
                pad_inches=0, 
                dpi=75)  # Lower DPI = smaller image file
    
    plt.close()  # Close the figure to free memory
    
    # Get the actual image size
    img = Image.open(output_path)
    print(f"Saved image dimensions: {img.size[0]} x {img.size[1]} pixels")
    print(f"Saved clean mel spectrogram as {output_path}")
    
    return mel_spect_db