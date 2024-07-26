import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tkinter import Tk
from tkinter.filedialog import askopenfilename


faces_data_dir = pathlib.Path('Faces')

batch_size = 32
img_height = 224
img_width = 224


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    faces_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    faces_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

class_names = train_generator.class_indices
print("\nClass Names:", class_names)


base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 28
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)


def predict_image(model, img_path, img_height, img_width, class_names):
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = tf.nn.softmax(predictions[0])

    return predicted_class[0], 100 * np.max(confidence), img

def select_image():
    Tk().withdraw()  
    file_path = askopenfilename(title='Select an image', filetypes=[('Image files', '*.jpg *.jpeg *.png')])
    return file_path


while True:
    image_path = select_image()
    if not image_path: 
        break
    predicted_class, confidence, img = predict_image(model, image_path, img_height, img_width, list(class_names.keys()))


    plt.imshow(img)
    plt.title(f'Predicted: {list(class_names.keys())[predicted_class]} with {confidence:.2f}% confidence')
    plt.axis('off')
    plt.show()  

    