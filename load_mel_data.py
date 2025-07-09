import os
import cv2
import numpy as np
from pre_process_audio import generate_mel_spectrogram
import matplotlib.pyplot as plt


GENRE_TO_INDEX = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
}




def load_my_gatzan_data():
    """
    Loads the GTZAN dataset from .npy files using a robust, dynamic path.
    """
    # Get the absolute path of the script's directory for robust pathing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Correctly point to the directory containing the .npy files
    data_path = os.path.join(script_dir, "Data", "my_gatzan_img_npy_data")

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"[Error] The directory {data_path} was not found. Please ensure your data is processed and in the correct location.")

    print(f"[INFO] Loading mel data from: {data_path}")

    X_data = []
    Y_data = []

    # Get a list of all genre subdirectories
    genres = [g for g in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, g))]

    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        y_label = GENRE_TO_INDEX.get(genre)

        if y_label is None:
            print(f"Warning: Genre '{genre}' not found in GENRE_TO_INDEX. Skipping.")
            continue

        for filename in sorted(os.listdir(genre_path)):
            if filename.endswith(".npy"):
                file_path = os.path.join(genre_path, filename)
                data = np.load(file_path)
                X_data.append(data)
                Y_data.append(y_label)

    # Find the maximum width (time steps) among all spectrograms
    max_width = 0
    if X_data:
        max_width = max(spectrogram.shape[1] for spectrogram in X_data)

    # Pad each spectrogram to the max_width
    X_padded_data = []
    for spectrogram in X_data:
        pad_width = max_width - spectrogram.shape[1]
        # Pad only the second axis (width) on the right side
        padded_spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        X_padded_data.append(padded_spectrogram)

    # Convert lists to numpy arrays
    X_data = np.array(X_padded_data)
    Y_data = np.array(Y_data)

    return X_data, Y_data




def load_data_nexus():
    """
    Presents a menu to the user and loads the selected dataset.
    """
    print("Please choose an option:")
    print("1) Load GTZAN .npy data")
    print("2) Load other data")

    choice = input("Enter your choice: ")

    if choice == '1':
        print("Loading GTZAN data from directory...")
        # Directly call the function and return its result
        return load_my_gatzan_data()
    elif choice == '2':
        print("Loading other data... (Not implemented yet)")
        return None, None
    else:
        print("Invalid choice.")
        return None, None

