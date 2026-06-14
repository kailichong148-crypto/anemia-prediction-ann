import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 2. Load dataset
data = pd.read_csv("anemia.csv")

print("Original dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# 3. Basic dataset checking
print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

print("\nClass distribution before removing duplicates:")
print(data["Result"].value_counts())

print("\nNumber of duplicate rows:", data.duplicated().sum())

# 4. Remove duplicate rows
data = data.drop_duplicates()

print("\nDataset shape after removing duplicates:", data.shape)

print("\nClass distribution after removing duplicates:")
print(data["Result"].value_counts())

# 5. Select input features and target output
X = data[["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]]
y = data["Result"]

# 6. Split dataset into training, validation, and testing sets
#    70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set:", X_test.shape)

# 7. Normalize input features
#    Important: fit scaler only on training data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 8. Build ANN model
model = Sequential()

# Hidden layer 1
model.add(Dense(
    16,
    activation="relu",
    input_shape=(5,)
))

# Dropout layer to reduce overfitting
model.add(Dropout(0.2))

# Hidden layer 2
model.add(Dense(
    8,
    activation="relu"
))

# Output layer
model.add(Dense(
    1,
    activation="sigmoid"
))

# 9. Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()

# 10. Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# 11. Train model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# 12. Evaluate model using testing data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n==================== Final Test Result ====================")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# 13. Make predictions
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

# 14. Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-Anemic", "Anemic"]
))

auc_score = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", auc_score)

# 15. Plot training accuracy and validation accuracy
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_graph.png")
plt.show()

# 16. Plot training loss and validation loss
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_graph.png")
plt.show()

# 17. Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve, AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png")
plt.show()

# 18. Save trained model
model.save("anemia_ann_model.h5")

print("\nModel saved as anemia_ann_model.h5")
print("Accuracy graph saved as accuracy_graph.png")
print("Loss graph saved as loss_graph.png")
print("ROC curve saved as roc_curve.png")












































