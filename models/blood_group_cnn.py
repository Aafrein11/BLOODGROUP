from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

# ========== DATA LOADING & PREPROCESSING ==========
def load_dataset(data_path):
    images, labels = [], []
    class_labels = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 
                    'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}

    for label in class_labels:
        class_dir = os.path.join(data_path, label)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224)) / 255.0
            images.append(img)
            labels.append(class_labels[label])

    images = np.array(images)
    labels = np.array(labels)

    # One-hot encoding of labels
    labels = to_categorical(labels, num_classes=8)

    return images, labels

# Load Dataset
data_path = 'C:\\Users\\jenif\\Desktop\\blood-group-predictor\\dataset_blood_group'
x_data, y_data = load_dataset(data_path)

# Train-Validation Split (80% Training, 20% Validation)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# ========== DATA AUGMENTATION ==========
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ========== MODEL ARCHITECTURE (EfficientNetB3) ==========
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze base layers for transfer learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(8, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# ========== MODEL COMPILATION ==========
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ========== CLASS WEIGHTS FOR BALANCED TRAINING ==========
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), 
                                     y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

# ========== CALLBACKS ==========
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('models/blood_group_cnn_v6.h5', save_best_only=True, monitor='val_accuracy', mode='max')
]

# ========== MODEL TRAINING ==========
model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_val, y_val),
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ========== MODEL SAVING ==========
model.save('models/blood_group_cnn.h5')

print("✅ Model trained successfully with improved accuracy!")
