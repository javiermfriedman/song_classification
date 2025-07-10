import os
import sys
import numpy as np
import sklearn.model_selection as sk
from sklearn.model_selection import train_test_split

# Add current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pre_process_audio import audio_processing_nexus
from get_audio_data import get_data_nexus
from load_mel_data import load_data_nexus
from cnn_model import model_builder_nexus, plot_history # Import plot_history as well
from compile_train_model import compile_train_model_nexus # Import the new function
from augment_data import augment_data_nexus


def main():
    """
    Main function to run the audio processing and model training pipeline.
    """
    while True:
        print("\nPlease choose an option:")
        print("1) Get new data")
        print("2) Turn current audio into mel spectrogram data")
        print("3) Augment current data")
        print("4) Create and train a model")
        print("5) Predict with model")
        print("6) Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            print("Getting data...")
            get_data_nexus()
        elif choice == '2':
            print("Starting audio to mel spectrogram conversion...")
            audio_processing_nexus()
            print("Audio to mel spectrogram conversion finished.")
        elif choice == '3':
            print("Augmenting data...")
            augment_data_nexus()
            print("Data augmentation finished.")
        elif choice == '4':
            print("loading data")
            X_data, y_data = load_data_nexus()
            
            if X_data is None or y_data is None:
                print("Data loading cancelled or failed. Returning to main menu.")
                continue

            # Add channel dimension for Conv2D, because conv need extra channel 1 for this 3 for image 
            if X_data.ndim == 3:
                X_data = np.expand_dims(X_data, axis=-1) # Add a new axis at the end

            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.25, random_state=42)
            print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

            # input_shape for the CNN model
            input_shape = (X_data.shape[1], X_data.shape[2], X_data.shape[3])
            print(f"Input shape for CNN: {input_shape}")

            model = model_builder_nexus(input_shape)

            print("Model built successfully.")
            # Compile and train the model using the nexus function
            history = compile_train_model_nexus(model, X_train, y_train, X_val, y_val)
            
            plot_history(history)

        elif choice == '5':
            print("Predicting with model... (Not implemented yet)")
        elif choice == '6':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
