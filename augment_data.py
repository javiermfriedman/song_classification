
import os
import numpy as np
import librosa
import soundfile as sf

def add_noise(y, noise_factor=0.005):
    """Adds random noise to an audio signal."""
    noise = np.random.randn(len(y))
    data_noise = y + noise_factor * noise
    return data_noise

def time_shift(y, sr, shift_max_sec=2):
    """Shifts the audio signal in time."""
    shift = np.random.randint(sr * shift_max_sec)
    if shift > 0:
        data_shift = np.pad(y, (shift, 0), mode='constant')[:len(y)]
    else:
        data_shift = np.pad(y, (0, -shift), mode='constant')[len(y):]
    return data_shift

def change_pitch_and_speed(y, sr, pitch_factor=0.7):
    """Changes the pitch and speed of the audio signal."""
    return librosa.effects.time_stretch(librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_factor), rate=pitch_factor)

def augment_data_nexus():
    """
    Augments audio data by creating variations of original files.
    """
    source_dir = os.path.join("Data", "genres_original")
    
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found.")
        return

    for genre_folder in os.listdir(source_dir):
        genre_path = os.path.join(source_dir, genre_folder)
        if not os.path.isdir(genre_path):
            continue

        print(f"Augmenting data for genre: {genre_folder}")

        for filename in os.listdir(genre_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(genre_path, filename)
                y, sr = librosa.load(file_path, sr=None)

                # 1. Add Noise
                y_noise = add_noise(y)
                sf.write(os.path.join(genre_path, f"{os.path.splitext(filename)[0]}_noise.wav"), y_noise, sr)

                # 2. Time Shift
                y_shifted = time_shift(y, sr)
                sf.write(os.path.join(genre_path, f"{os.path.splitext(filename)[0]}_shifted.wav"), y_shifted, sr)

                # 3. Change Pitch and Speed
                y_pitched_sped = change_pitch_and_speed(y, sr)
                sf.write(os.path.join(genre_path, f"{os.path.splitext(filename)[0]}_pitched_sped.wav"), y_pitched_sped, sr)

    print("Data augmentation complete.")

if __name__ == '__main__':
    augment_data_nexus()
