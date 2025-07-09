import os
import sys

from pre_process_audio import audio_processing_nexus



def main():
    """
    Main function to run the audio processing and model training pipeline.
    """
    while True:
        print("\nPlease choose an option:")
        print("1) Get new data")
        print("2) Turn current audio into mel spectrogram data")
        print("3) Create and train a model")
        print("4) Predict with model")
        print("5) Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            print("Getting new data... (Not implemented yet)")
        elif choice == '2':
            print("Starting audio to mel spectrogram conversion...")
            audio_processing_nexus()
            print("Audio to mel spectrogram conversion finished.")
        elif choice == '3':
            print("Creating and training model... (Not implemented yet)")
        elif choice == '4':
            print("Predicting with model... (Not implemented yet)")
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()