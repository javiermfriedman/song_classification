import os
import cv2
import numpy as np
from pre_process_audio import generate_mel_spectrogram
import matplotlib.pyplot as plt


# get

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


"""
    This function will take the raw audio data from genres_original and apply
    mel spectrum transformation on it and put the images into my_gatzan_img_data
"""
def make_gatzan_img_data():
    start_directory = "/Users/javierfriedman/Desktop/CodingStuff/AI/song_classification/Data/genres_original"
    output_directory = "/Users/javierfriedman/Desktop/CodingStuff/AI/song_classification/Data/my_gatzan_img_data"
    
    # Create the main output directory
    os.makedirs(output_directory, exist_ok=True)
    print(f"Created main directory: {output_directory}")

    # Track progress
    total_processed = 0
    total_errors = 0

    for dirpath, dirnames, filenames in os.walk(start_directory):

        if dirpath == start_directory: # skip the root directory
            continue

        # create new genre dir for output
        genre_name = os.path.basename(dirpath)

        genre_output_dir = os.path.join(output_directory, genre_name)  # FIXED: use genre_name
        os.makedirs(genre_output_dir, exist_ok=True)
        print(f"Created genre directory: {genre_output_dir}")

        # Loop through files in the current genre directory
        for song_name in filenames:

            if song_name.endswith(".wav"):
                try:
                    # Input path
                    song_path = os.path.join(dirpath, song_name)
                    
                    # Output path - FIXED: proper .png filename
                    output_filename = song_name.replace(".wav", ".png")
                    out_song_path = os.path.join(genre_output_dir, output_filename)

                    print(f"  Processing: {song_name}")
                    generate_mel_spectrogram(song_path, out_song_path)  # FIXED: no f-string
                    
                    total_processed += 1
                    
                except Exception as e:
                    print(f"  ERROR processing {song_name}: {str(e)}")
                    total_errors += 1

    print(f"\nProcessed {total_processed} files successfully, {total_errors} errors")

def print_img(img):
    plt.figure(figsize=(10, 4))
    plt.imshow(img)
    plt.title("Processed imaged")
    plt.axis('off')
    plt.show()
   
def get_my_gatzan_training_data():

    parent_dir = "/Users/javierfriedman/Desktop/CodingStuff/AI/song_classification/Data/my_gatzan_img_data/"
    print("[INFO] Loading mel data from:", parent_dir)

    X_data = []
    Y_data = []
    
    # Get all genre directories in parent directory
    for genre_dir in os.listdir(parent_dir):

        genre_path = os.path.join(parent_dir, genre_dir)
        # Skip if not a directory
        if not os.path.isdir(genre_path):
            print("skipping over non directory")
            continue

        # add the corresponding y label for given mel image
        y_label = GENRE_TO_INDEX.get(genre_dir)
        
        for filename in os.listdir(genre_path):
            if filename.endswith(".png"):
                file_path = os.path.join(genre_path, filename)
                # print(f"processing filepath: {file_path}")

                # print(f"appending y label: {y_label}\n\n")
                Y_data.append(y_label)

                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                X_data.append(img)

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    return X_data, Y_data



                
                


# def pre_process_mel_img(file_path);
    


def load_all_data():
    parent_dir = "/Users/javierfriedman/Desktop/CodingStuff/AI/song_classification/Data/genres_original/"

    matrix = {}
    
    # Get all genre directories in parent directory
    for genre_dir in os.listdir(parent_dir):
        genre_path = os.path.join(parent_dir, genre_dir)
        
        # Skip if not a directory
        if not os.path.isdir(genre_path):
            print("skipping over non directory")
            continue
            
        
        #images = []
        for filename in os.listdir(genre_path):
            if filename.endswith(".png"):
                file_path = os.path.join(genre_path, filename)

                print(f"processing filepath: {file_path}")
                
                # Generate spectrogram and add to list
                # spectrogram = generate_mel_spectrogram(file_path)
                # images.append(spectrogram)
                
                # print(f"Processed: {file_path}")
        
        # Add this genre's images as a column in the matrix
        # matrix[genre_dir] = images






