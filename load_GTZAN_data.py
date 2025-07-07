import os
from pre_process_audio import generate_mel_spectrogram
# get

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






