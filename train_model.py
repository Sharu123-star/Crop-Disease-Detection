import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Image settings
IMG_SIZE = (224, 224)

# Small batch size for our laptop
BATCH_SIZE = 8

# Number of disease/health classes
NUM_CLASSES = 4

print("TensorFlow version:", tf.__version__)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

# Load the CSV files
train_df = pd.read_csv("splits/train.csv")
val_df = pd.read_csv("splits/validation.csv")
test_df = pd.read_csv("splits/test.csv")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label

    # Convert file paths and labels into TensorFlow datasets

class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight"
]

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

train_labels = train_df["label"].map(class_to_index).values
val_labels = val_df["label"].map(class_to_index).values
test_labels = test_df["label"].map(class_to_index).values

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df["filepath"].values, train_labels)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df["filepath"].values, val_labels)
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_df["filepath"].values, test_labels)
)

train_ds = train_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("TensorFlow datasets created successfully!")

# Test one batch
for images, labels in train_ds.take(1):
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Batch loaded successfully!")

    # Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained layers
base_model.trainable = False

print("MobileNetV2 loaded successfully!")
print("Number of layers:", len(base_model.layers))

# Build the complete classification model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

print("Classification model created successfully!")
model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")

# Check the model
print("\nModel is ready for training!")
print("Trainable parameters:", model.count_params())

# Train the classification layer
EPOCHS = 5

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("Training completed!")

# Evaluate the trained model on unseen test data
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nFinal Test Results")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Get predictions for the test set
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(tf.argmax(predictions, axis=1).numpy())

# Classification report
print("\nClassification Report")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix")
print(cm)

# Display confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix - Tomato Disease Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(NUM_CLASSES), class_names, rotation=45)
plt.yticks(range(NUM_CLASSES), class_names)

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# Save the trained model
model.save("crop_disease_mobilenetv2.keras")

print("\nModel saved successfully!")