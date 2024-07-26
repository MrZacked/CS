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


animals_data_dir = pathlib.Path('animals')
furniture_data_dir = pathlib.Path('images')
faces_data_dir=pathlib.Path('Faces')


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


#copy_images(flowers_data_dir, combined_data_dir)
#copy_images(animals_data_dir, combined_data_dir)
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
    seed=321,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print("\nClass Names:", class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(len(train_ds)).prefetch(buffer_size=AUTOTUNE)
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
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

selected_images_info = []

def display_and_select_images(data, num_images_to_show=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    num_batches_to_take = 5
    images = []
    labels = []
    
    for images_batch, labels_batch in data.shuffle(buffer_size=len(data)).take(num_batches_to_take):
        images.extend(images_batch.numpy())
        labels.extend(labels_batch.numpy())
    
    images = images[:num_images_to_show]
    labels = labels[:num_images_to_show]
    axes = axes.flatten()


    selected_indices = set()

    def onclick(event):
        for i, ax in enumerate(axes):
            if event.inaxes == ax:
                if i not in selected_indices:
                    selected_indices.add(i)
                    selected_images_info.append({
                        'image': images[i],
                        'index': i,
                        'label': labels[i]
                    })
                    ax.set_title(f"Selected {i}")
                    fig.canvas.draw()
                if len(selected_images_info) >= 5:
                    plt.close(fig)
                break

    for i in range(num_images_to_show):
        ax = axes[i]
        ax.imshow(images[i].astype("uint8"))
        ax.set_title(f"Image {i}")
        ax.axis("off")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return images, selected_indices

def get_indices_from_user(num_images_to_select=5):
    selected_indices = set()
    while len(selected_indices) < num_images_to_select:
        try:
            idx = int(input(f"Enter index of image to select (0-9, remaining {num_images_to_select - len(selected_indices)}): "))
            if idx in range(10) and idx not in selected_indices:
                selected_indices.add(idx)
            else:
                print("Invalid or duplicate index.")
        except ValueError:
            print("Please enter a valid integer.")
    return selected_indices

def predict_images():
    for info in selected_images_info:
        image = info['image']
        img_array = tf.expand_dims(image, 0)  
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        info['predicted_class'] = predicted_class
        info['confidence'] = confidence


images, selected_indices = display_and_select_images(train_ds)

if len(selected_images_info) < 5:
    print("Not enough images selected by clicking. Please select more images using indices.")
    additional_indices = get_indices_from_user(5 - len(selected_images_info))
    for idx in additional_indices:
        selected_images_info.append({
            'image': images[idx],
            'index': idx
        })


predict_images()


for info in selected_images_info:
    plt.imshow(info['image'].astype("uint8"))
    plt.title(f"Predicted: {info['predicted_class']} ({info['confidence']:.2f}%)")
    plt.axis("off")
    plt.show()
    print(f"Index: {info['index']} - Predicted Class: {info['predicted_class']} with {info['confidence']:.2f}% confidence.")





