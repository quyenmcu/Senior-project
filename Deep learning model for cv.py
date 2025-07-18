import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0,DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths to your dataset
train_dir = r"D:\Senior project\dataset_split\train"
val_dir = r"D:\Senior project\dataset_split\val"
test_dir = r"D:\Senior project\dataset_split\test"

# Image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Preprocessing function for images with enhanced augmentation
def preprocess_image_with_augmentation(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0  # Normalize pixel values to [0, 1]

    # Enhanced data augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)

    return image, label

# Preprocessing function without augmentation
def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return image, label

# Create datasets
def create_dataset(directory, augment=False):
    class_names = sorted(os.listdir(directory))
    num_classes = len(class_names)

    file_paths = []
    labels = []

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        file_paths.extend(files)
        labels.extend([idx] * len(files))

    file_paths = tf.constant(file_paths)
    labels = tf.one_hot(labels, num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_image_with_augmentation if augment else preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, num_classes

# Create train, validation, and test datasets
train_dataset, num_classes = create_dataset(train_dir, augment=True)
val_dataset, _ = create_dataset(val_dir)
test_dataset, _ = create_dataset(test_dir)

# Compute class weights for imbalanced datasets
train_labels = [tf.argmax(label).numpy() for _, label in train_dataset.unbatch()]
class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

# Model training function
def build_and_train_model(base_model_fn, model_name):
    # Load base model
    base_model = base_model_fn(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base model

    # Add custom layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)  # Dropout for regularization
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        class_weight=class_weights,
        callbacks=[early_stop, lr_scheduler]
    )

    # Fine-tune the base model
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    history_fine_tune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        class_weight=class_weights,
        callbacks=[early_stop, lr_scheduler]
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_dataset)
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"{model_name} Results:")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the model
    model.save(f"{model_name.lower()}_finetuned.keras")
    print(f"{model_name} model saved as '{model_name.lower()}_finetuned.keras'.")

    return model, history, val_accuracy, test_accuracy

# Train MobileNetV2
mobilenet_model, mobilenet_history, mobilenet_val_acc, mobilenet_test_acc = build_and_train_model(MobileNetV2, "MobileNetV2")

# Train EfficientNetB0
efficientnet_model, efficientnet_history, efficientnet_val_acc, efficientnet_test_acc = build_and_train_model(EfficientNetB0, "EfficientNetB0")

# Train DenseNet121
densenet_model, densenet_history, densenet_val_acc, densenet_test_acc = build_and_train_model(DenseNet121, "DenseNet121")

# Compare Results
print("\nModel Performance Comparison:")
print(f"MobileNetV2  - Validation Accuracy: {mobilenet_val_acc:.4f}, Test Accuracy: {mobilenet_test_acc:.4f}")
print(f"EfficientNetB0 - Validation Accuracy: {efficientnet_val_acc:.4f}, Test Accuracy: {efficientnet_test_acc:.4f}")
print(f"DenseNet121 - Validation Accuracy: {densenet_val_acc:.4f}, Test Accuracy: {densenet_test_acc:.4f}")

# Plot Training Histories
def plot_training_history(history, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

# Plot all model training histories
plot_training_history(mobilenet_history, "MobileNetV2")
plot_training_history(efficientnet_history, "EfficientNetB0")
plot_training_history(densenet_history, "DenseNet121")