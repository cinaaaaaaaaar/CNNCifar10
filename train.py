import tensorflow as tf
from classes.Model import Model
from preprocess import train_images, train_labels, test_images, test_labels

model = Model()
model.create_model()
model.train_network(train_images, train_labels, test_images, test_labels,
                    epochs=8)
