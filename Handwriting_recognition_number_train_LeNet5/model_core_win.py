import keras
import numpy as np
from PIL import Image


class LeNet5Custom:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.model = self._create_model()


    def _create_model(self):
        model = keras.models.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),  # 明确指定输入形状
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Flatten(),

            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(self.dropout_rate),

            keras.layers.Dense(10, activation='softmax')
        ])
        return model


    def compile_model(self):
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model


    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Dropout):
                layer.rate = self.dropout_rate
        return self.model


    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array[img_array != 255] = 0 # 将像素值非255的全部设为0
        img_array = img_array.astype("float32") / 255
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array


    def predict(self, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(preprocessed_image)
        return predictions
