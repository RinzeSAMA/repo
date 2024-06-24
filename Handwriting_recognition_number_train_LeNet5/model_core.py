import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image


class LeNet5Custom:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.model = self._create_model()


    def _create_model(self):
        model = Sequential([
            Input(shape=(28, 28, 1)),  # 明确指定输入形状
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Flatten(),

            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate),

            Dense(10, activation='softmax')
        ])
        return model


    def compile_model(self):
        self.model.compile(optimizer=Adam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model


    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                layer.rate = self.dropout_rate
        return self.model


    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array[img_array != 255] = 0  # 将像素值非255的全部设为0
        img_array = img_array.astype("float32") / 255
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array


    def predict(self, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(preprocessed_image)
        return predictions
