import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt


# Generate training data
def generate_data(num_samples):
    X = np.random.uniform(-10, 10, num_samples)
    y = X ** 3 + X ** 2 + 10
    return X, y


# Generate training and validation data
train_X, train_y = generate_data(1000)
val_X, val_y = generate_data(200) # train_X[-200:], train_y[-200:]

# Create a neural network model
model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train the model
history = model.fit(train_X, train_y, epochs=500, batch_size=32, validation_data=(val_X, val_y))

# Generate new input data for prediction
new_inputs = np.random.uniform(-10, 10, 100).reshape(-1, 1)
new_outputs = model.predict(new_inputs)

# Sort the inputs and corresponding outputs for plotting
sorted_indices = np.argsort(new_inputs, axis=0).flatten()
sorted_inputs = new_inputs[sorted_indices]
sorted_outputs = new_outputs[sorted_indices]

# Expected outputs for new inputs
expected_outputs = sorted_inputs ** 3 + sorted_inputs ** 2 + 10

# Plot the results
plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot expected vs predicted outputs
plt.subplot(1, 2, 2)
plt.scatter(sorted_inputs, expected_outputs, label='Expected Output', color='blue')
plt.scatter(sorted_inputs, sorted_outputs, label='Predicted Output', color='red')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Expected vs Predicted Output')
plt.legend()

plt.tight_layout()
plt.show()

# Print the new inputs and the predicted outputs
print("New Inputs: ", new_inputs.flatten())
print("Predicted Outputs: ", new_outputs.flatten())
