import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pathlib
import shutil
import os

from tensorflow import keras
from keras.models import Sequential
from keras import layers


flowers_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
flowers_data_dir = tf.keras.utils.get_file('flower_photos', origin=flowers_dataset_url, untar=True)
flowers_data_dir = pathlib.Path(flowers_data_dir)


animals_data_dir = pathlib.Path('animals')
furniture_data_dir = pathlib.Path('images')


combined_data_dir = pathlib.Path('/tmp/combined_photos')
if combined_data_dir.exists():
    shutil.rmtree(combined_data_dir)  # Remove previous content if any
combined_data_dir.mkdir(parents=True, exist_ok=True)


for item in flowers_data_dir.glob('*/*'):
    target_dir = combined_data_dir / item.parent.name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(item, target_dir / item.name)


for item in animals_data_dir.glob('*/*'):
    target_dir = combined_data_dir / item.parent.name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(item, target_dir / item.name)


furniture_target_dir = combined_data_dir / 'furniture'
furniture_target_dir.mkdir(parents=True, exist_ok=True)
for item in furniture_data_dir.glob('*.jpg'):
    shutil.copy(item, furniture_target_dir / item.name)


batch_size = 32
img_height = 180
img_width = 180


train_ds = tf.keras.utils.image_dataset_from_directory(
    combined_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    combined_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print("Class Names:", class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs = 11
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

selected_images_info = []

def display_and_select_images(data, num_images_to_show=10, num_images_to_select=5):
    plt.figure(figsize=(15, 10))
    images_batch, _ = next(iter(data.shuffle(buffer_size=1000).take(1)))
    images = images_batch[:num_images_to_show]
    
    for i in range(num_images_to_show):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()
    
    indexes = input(f"Select {num_images_to_select} image indexes separated by spaces (0 to {num_images_to_show - 1}): ")
    selected_indexes = list(map(int, indexes.split()))
    
    for index in selected_indexes:
        selected_images_info.append({
            'image': images[index],
                'index': index
            })


    def predict_images():
        for info in selected_images_info:
            image = info['image']
            img_array = tf.expand_dims(image, 0)  # Create a batch
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)
            info['predicted_class'] = predicted_class
            info['confidence'] = confidence


    display_and_select_images(train_ds)

predict_images()

for info in selected_images_info:
    plt.imshow(info['image'].numpy().astype("uint8"))
    plt.title(f"Predicted: {info['predicted_class']} ({info['confidence']:.2f}%)")
    plt.axis("off")
    plt.show()
    print(f"Index: {info['index']} - Predicted Class: {info['predicted_class']} with {info['confidence']:.2f}% confidence.")

