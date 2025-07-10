import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

def compile_train_model_nexus(model, X_train, y_train, X_val, y_val):
    """
    Provides a nexus for compiling and training a Keras model.
    Allows user to set training parameters or use defaults, choose loss function,
    and decide whether to save the trained model.

    Args:
        model (tf.keras.Model): The Keras model to compile and train.
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_val (np.array): Validation features.
        y_val (np.array): Validation labels.

    Returns:
        tf.keras.callbacks.History: The training history object, or None if training is skipped.
    """

    # Default parameters
    epochs = 16
    batch_size = 32
    learning_rate = 0.0001
    loss_function = sparse_categorical_crossentropy # Default loss
    metrics = ['accuracy']
    early_stopping_patience = 3

    print("\n--- Model Compilation and Training Setup ---")
    choice = input("Use default training parameters (epochs=16, batch_size=32, learning_rate=0.0001)? (y/n): ")

    if choice.lower() == 'n':
        try:
            epochs_input = input(f"Enter number of epochs (default: {epochs}): ")
            if epochs_input: epochs = int(epochs_input)

            batch_size_input = input(f"Enter batch size (default: {batch_size}): ")
            if batch_size_input: batch_size = int(batch_size_input)

            lr_input = input(f"Enter learning rate (default: {learning_rate}): ")
            if lr_input: learning_rate = float(lr_input)

        except ValueError:
            print("Invalid input. Using default training parameters.")

    # Loss function selection
    while True:
        print("\nChoose a loss function:")
        print("1) sparse_categorical_crossentropy (for integer labels)")
        print("2) categorical_crossentropy (for one-hot encoded labels)")
        loss_choice = input("Enter your choice (default: 1): ")

        if loss_choice == '1' or not loss_choice:
            loss_function = sparse_categorical_crossentropy
            break
        elif loss_choice == '2':
            loss_function = categorical_crossentropy
            break
        else:
            print("Invalid choice. Please select 1 or 2.")

    optimizer = Adam(learning_rate=learning_rate)

    # Compile model
    print("Compiling model...")
    model.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=metrics,
    )
    print("Model compiled successfully.")
    model.summary()

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1,
    )
    print("Model training finished.")

    # Save model
    save_choice = input("\nDo you want to save the trained model? (y/n): ")
    if save_choice.lower() == 'y':
        model_filename = input("Enter filename to save model (e.g., my_model.h5): ")
        if model_filename:
            try:
                model.save(model_filename)
                print(f"Model saved successfully as {model_filename}")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            print("Model not saved: No filename provided.")
    else:
        print("Model not saved.")

    return history