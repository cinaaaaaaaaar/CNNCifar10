import os
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.losses import SparseCategoricalCrossentropy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(10))
        model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(
            from_logits=True), metrics=["accuracy"])
        self.model = model
        return model

    def train_network(self, x_train, y_train, x_test, y_test, epochs):
        path = "weights\\weights.h5"
        self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        self.model.save_weights(path)
