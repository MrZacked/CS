import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import shutil

flowers_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
flowers_data_dir = tf.keras.utils.get_file('flower_photos', origin=flowers_dataset_url, untar=True)
flowers_data_dir = pathlib.Path(flowers_data_dir)

animals_data_dir=pathlib.Path('animals')
furniture_data_dir=pathlib.Path('images')
faces_data_dir = pathlib.Path('Faces')

combined_data_dir = pathlib.Path('/tmp/combined_photos')
if combined_data_dir.exists():
    shutil.rmtree(combined_data_dir)  
combined_data_dir.mkdir(parents=True, exist_ok=True)

def copy_images(src_dir, target_base_dir):
    for item in src_dir.glob('*/*'):
        target_dir = target_base_dir / item.parent.name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(item, target_dir / item.name)
    print(f"Copied images from {src_dir}")

# We are going to comment these of so we can focus on testing the Faces since our animals and flowers folder contains a larger amount of pictures.

# copy_images(flowers_data_dir, combined_data_dir)
# copy_images(animals_data_dir, combined_data_dir)

copy_images(faces_data_dir, combined_data_dir)

furniture_target_dir = combined_data_dir / 'furniture'
furniture_target_dir.mkdir(parents=True, exist_ok=True)
for item in furniture_data_dir.glob('*.jpg'):
    shutil.copy(item, furniture_target_dir / item.name)
print(f"Copied furniture images from {furniture_data_dir}")

print("\nCombined directory structure and image counts:")
for path in combined_data_dir.glob('*/'):
    count = len(list(path.glob('*')))
    print(f"{path.name}: {count} images")

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
print("\nClass Names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(buffer_size=1000).cache().prefetch(buffer_size=AUTOTUNE)
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

epochs = 16
history=model.fit(train_ds, validation_data=val_ds,epochs=epochs)

def display_and_select_images(data, num_images_to_show=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    image_list = []

    for images_batch, _ in data.shuffle(buffer_size=1000).take(5):
        image_list.extend(images_batch.numpy())

    np.random.shuffle(image_list)
    images = image_list[:num_images_to_show]
    axes = axes.flatten()

    pred_figs = []

    def onclick(event):
        for i, ax in enumerate(axes):
            if event.inaxes == ax:
                image = images[i]
                img_array = tf.expand_dims(image, 0)  # Create a batch
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                predicted_class = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                pred_fig = plt.figure()
                plt.imshow(image.astype("uint8"))
                plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
                plt.axis("off")
                pred_fig.show()
                pred_figs.append(pred_fig)
                break

    def on_close(event):
        for pred_fig in pred_figs:
            plt.close(pred_fig)

    def reset_images(event):
        nonlocal images
        image_list = []
        for images_batch, _ in data.shuffle(buffer_size=1000).take(5):
            image_list.extend(images_batch.numpy())
        np.random.shuffle(image_list)
        images = image_list[:num_images_to_show]
        for i in range(num_images_to_show):
            axes[i].imshow(images[i].astype("uint8"))
            axes[i].set_title(f"Image {i}")
            axes[i].axis("off")
        fig.canvas.draw()

    reset_button_ax = fig.add_axes([0.8, 0.01, 0.1, 0.05])               
    reset_button = plt.Button(reset_button_ax, 'Reset Images')
    reset_button.on_clicked(reset_images)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('close_event', on_close)

    for i in range(num_images_to_show):
        axes[i].imshow(images[i].astype("uint8"))
        axes[i].set_title(f"Image {i}")
        axes[i].axis("off")
    plt.show()

display_and_select_images(train_ds)





