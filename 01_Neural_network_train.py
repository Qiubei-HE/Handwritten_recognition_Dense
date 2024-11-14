############### Note 📒 ###############
## 08/11/2024

# one_hot_vector(): Input a number and return the corresponding one-hot vector.

# The professor explained the logic of model training: 
# the equation for calculating error (MSE), and the process of minimizing this error by adjusting the model parameters ('min L(x)')

# He also introduced the one-hot vector, 
# which can convert categorical variables into a form that algorithms can easily utilize (binary vectors).
#######################################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Parameters
epochs = 10
batch_size = 32
learning_rate = 0.001

# Load and prepare the dataset
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=16)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=16)
print('The label befor One-hot encode','\n',y_train)



# One-hot encode labels !!!!
encoder = OneHotEncoder(sparse_output=False, categories='auto')

# Reshape the labels to a 2D array (required by OneHotEncoder)
y_train = np.array(y_train).reshape(-1, 1)
y_eval = np.array(y_eval).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# Fit the encoder on y_train and transform y_train, y_eval, and y_test
y_train = encoder.fit_transform(y_train)
y_eval = encoder.transform(y_eval)
y_test = encoder.transform(y_test)

print('The label after One-hot encode','\n',y_train)

# Build the model
model = Sequential([
    Flatten(input_shape=(64,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model with multiple metrics
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                       tf.keras.metrics.Recall(name='recall')])

# Train the model and record metrics
history = model.fit(X_train, y_train, 
                    validation_data=(X_eval, y_eval), 
                    epochs=epochs, 
                    batch_size=batch_size)

# Calculate F1-score manually
precision = history.history['val_precision']
recall = history.history['val_recall']
f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]

# Plot training and evaluation loss
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'], label="Training Loss")
plt.plot(epochs_range, history.history['val_loss'], label="Evaluation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (Categorical Crossentropy)")
plt.title("Training and Evaluation Loss Over Epochs")
plt.legend()

# Plot validation metrics
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
plt.plot(epochs_range, precision, label="Validation Precision")
plt.plot(epochs_range, recall, label="Validation Recall")
plt.plot(epochs_range, f1_scores, label="Validation F1 Score")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.title("Validation Metrics Over Epochs")
plt.legend()

plt.tight_layout()
# Save the plot as an image file before displaying
plt.savefig("01_training_evaluation_metrics.png", format="png")
plt.show()

# Evaluate the model on the test set after training
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)

# Calculate the F1 score for the test set
if (test_precision + test_recall) > 0:
    test_f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
else:
    test_f1_score = 0

# Print the test set evaluation results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1_score:.4f}")
