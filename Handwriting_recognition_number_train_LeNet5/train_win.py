from datetime import datetime
import keras
from model_core import LeNet5Custom
import matplotlib.pyplot as plt

# 打印模型框架基本信息
print(keras.__version__)


def load_and_preprocess_data():
    # 加载MNIST数据集
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # 反转像素值：255 - x
    train_images = 255 - train_images
    test_images = 255 - test_images

    # 归一化像素值到 [0, 1] 区间
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    # 由于 MNIST 的图像是灰度图像，需要增加一个颜色通道
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    # 对标签进行分类编码
    train_labels = keras.utils.to_categorical(train_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 15:
        lr *= 0.1
    elif epoch > 55:
        lr *= 0.01
    return lr


class TrainingHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))



def train_model_dp(train_images, train_labels, test_images, test_labels, dropout_rates, epochs):
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=3,
                                                   restore_best_weights=True)
    history_callback = TrainingHistory()

    best_accuracy = 0.0
    best_dropout_rate = None
    best_model = None

    for dropout_rate in dropout_rates:
        print(f"Training model with dropout rate: {dropout_rate}")

        model = LeNet5Custom(dropout_rate)
        model = model.compile_model()

        model.summary()

        model.fit(train_images,
                  train_labels,
                  epochs=epochs,
                  batch_size=256,
                  validation_data=(test_images, test_labels),
                  callbacks=[lr_scheduler, early_stopping, history_callback])

        # 评估模型
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Test Accuracy with dropout rate {dropout_rate}: {test_acc:.4f} with loss {test_loss:.4f}')

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_dropout_rate = dropout_rate
            best_model = model

    return best_model, best_dropout_rate, best_accuracy, history_callback.history


def plot_history(history):
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_model(model, accuracy):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳并格式化为字符串
    filepath = f'./model/lenet5_model_best_{timestamp}.h5'
    model.save(filepath=filepath)
    print(f"Best model weights saved to {filepath} with test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    keras.backend.clear_session()

    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

    dropout_rates = [0.3]  # 0.3 is the best of [0.1, 0.2, 0.3, 0.4, 0.5]
    epochs = 100

    best_model, best_dropout_rate, best_accuracy, history = train_model_dp(train_images,
                                                                           train_labels,
                                                                           test_images,
                                                                           test_labels,
                                                                           dropout_rates,
                                                                           epochs)

    print(f'Best dropout rate: {best_dropout_rate} with test accuracy: {best_accuracy:.4f}')

    save_model(best_model, best_accuracy)
    plot_history(history)