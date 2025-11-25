https://colab.research.google.com/drive/1c3PAtg54sKbgdf4Sp6Enc001U0xT0iL2?usp=sharing
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import matplotlib.pyplot as plt
import zipfile

# =====================================================
# 🔧 Step 1: Mount Google Drive (optional if you use uploads)
# =====================================================
from google.colab import drive
drive.mount('/content/drive')

# =====================================================
# 📦 Step 2: Unzip your uploaded datasets
# =====================================================
!cp -r "/content/drive/MyDrive/dataset_split" "/content/dataset_split_local"
train_dir = "/content/drive/MyDrive/dataset_split/train"
val_dir   = "/content/drive/MyDrive/dataset_split/val"
test_dir  = "/content/drive/MyDrive/dataset_split/test"
# =====================================================
# ⚙️ Step 3: Define parameters
# =====================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# =====================================================
# 🧩 Step 4: Preprocessing and augmentation
# =====================================================
def preprocess_image_with_augmentation(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    return image, label

def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return image, label

# =====================================================
# 🧠 Step 5: Create dataset function
# =====================================================
def create_dataset(directory, augment=False):
    class_names = sorted(os.listdir(directory))
    num_classes = len(class_names)

    file_paths, labels = [], []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        files = [os.path.join(class_dir, f)
                 for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        file_paths.extend(files)
        labels.extend([idx] * len(files))

    file_paths = tf.constant(file_paths)
    labels = tf.one_hot(labels, num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_image_with_augmentation if augment else preprocess_image,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset, num_classes

train_dataset, num_classes = create_dataset(train_dir, augment=True)
val_dataset, _ = create_dataset(val_dir)
test_dataset, _ = create_dataset(test_dir)

# =====================================================
# ⚖️ Step 6: Compute class weights
# =====================================================
train_labels = [tf.argmax(label).numpy() for _, label in train_dataset.unbatch()]
class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# =====================================================
# 🧠 Step 7: Build + train model function
# =====================================================
def build_and_train_model(base_model_fn, model_name):
    base_model = base_model_fn(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=20, class_weight=class_weights,
                        callbacks=[early_stop, lr_scheduler])

    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history_fine = model.fit(train_dataset, validation_data=val_dataset,
                             epochs=10, class_weight=class_weights,
                             callbacks=[early_stop, lr_scheduler])

    val_loss, val_acc = model.evaluate(val_dataset)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"{model_name} | Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    model.save(f"/content/{model_name.lower()}_finetuned.keras")
    print(f"Saved {model_name.lower()}_finetuned.keras")
    return model, history

# =====================================================
# 🧪 Step 8: Train Models
# =====================================================
mobilenet_model, mobilenet_history = build_and_train_model(MobileNetV2, "MobileNetV2")
efficientnet_model, efficientnet_history = build_and_train_model(EfficientNetB0, "EfficientNetB0")
densenet_model, densenet_history = build_and_train_model(DenseNet121, "DenseNet121")

# =====================================================
# 📊 Step 9: Plot results
# =====================================================
def plot_training_history(history, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_training_history(mobilenet_history, "MobileNetV2")
plot_training_history(efficientnet_history, "EfficientNetB0")
plot_training_history(densenet_history, "DenseNet121")
