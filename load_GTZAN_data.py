import os
from pre_process_audio import generate_mel_spectrogram
# get

def load_all_data():
    start_directory = "/Users/javierfriedman/Desktop/CodingStuff/AI/song_classification/Data/genres_original"

    data = {}

    for dirpath, dirnames, filenames in os.walk(start_directory):
        # print(f"Current directory: {dirpath}")
        genre = os.path.basename(dirpath)
        songs = []

        # Loop through subdirectories in the current directory
        for filename in filenames:
            test_audio_path = "blues.00000.wav"

            if filename.endswith(".wav") and (filename == test_audio_path):
                song_path = os.path.join(dirpath, filename)

                print(f"  filename: {os.path.join(dirpath, filename)}")
                generate_mel_spectrogram(song_path)
                songs.append(song_path)
                # preprocess the filename and append image to df

        data[genre] = songs

    return data




