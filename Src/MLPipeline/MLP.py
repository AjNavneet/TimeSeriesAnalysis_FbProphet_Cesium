from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

# Define a class for the MLP model
class MLP:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        # Hyperparameters
        epochs = 1000
        lr = 0.05

        # Initialize the model
        adam = optimizers.Adam(lr)
        model_mlp = Sequential()

        # Add layers to the model
        model_mlp.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
        model_mlp.add(Dense(1))  # Output layer

        # Compile the model
        model_mlp.compile(loss='mse', optimizer=adam)
        model_mlp.summary()

        # Train the model
        mlp_history = model_mlp.fit(X_train, Y_train, epochs=epochs, verbose=2)

        # Make predictions on the test data
        mlp_pred = model_mlp.predict(X_test)

        # Save the trained model to a file
        model_mlp.save("../Output/models/model_cesium.h5")

        # Plot the predictions and true values
        plt.figure(figsize=(20, 5))
        plt.plot(mlp_pred)
        plt.plot(Y_test.values, color='red')
        plt.savefig("../Output/plots/accuracy.png")
