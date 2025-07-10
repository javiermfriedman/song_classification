import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers, Sequential

def build_model_1(input_shape):
    """Generates CNN model
    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def build_model_2(input_shape):
    model = Sequential()

    # 1st conv block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # 2nd conv block
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # 3rd conv block
    model.add(layers.Conv2D(128, (2, 2), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    # replace Flatten with GlobalAveragePooling
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))

    return model

def build_model_3(input_shape, dropout_rate_1=0.3, dropout_rate_2=0.3, dropout_rate_3=0.4, dropout_rate_dense=0.5):
    model = Sequential()

    # 1st conv block
    model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate_1))

    # 2nd conv block
    model.add(layers.Conv2D(512, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate_2))

    # 3rd conv block
    model.add(layers.Conv2D(1024, (2, 2), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate_3))

    # replace Flatten with GlobalAveragePooling
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu')) # Increased dense layer neurons
    model.add(layers.Dropout(dropout_rate_dense))

    model.add(layers.Dense(10, activation='softmax'))

    return model

def model_builder_nexus(input_shape):
    """
    Prompts the user to select a model architecture and returns the chosen model.

    Args:
        input_shape (tuple): The input shape for the model.

    Returns:
        keras.Model: The selected and built Keras model.
    """
    while True:
        print("\nPlease choose a model architecture:")
        print("1) Model 1 (Simpler CNN)")
        print("2) Model 2 (CNN with L2 regularization and GlobalAveragePooling)")
        print("3) Model 3 (Increased Capacity CNN with L2 regularization and GlobalAveragePooling)")
        print("4) Exit model selection")

        choice = input("Enter your choice: ")

        if choice == '1':
            print("Building Model 1...")
            return build_model_1(input_shape)
        elif choice == '2':
            print("Building Model 2...")
            return build_model_2(input_shape)
        elif choice == '3':
            print("Building Model 3...")
            return build_model_3(input_shape)
        elif choice == '4':
            print("Exiting model selection.")
            return None
        else:
            print("Invalid choice. Please try again.")


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    plt.show()