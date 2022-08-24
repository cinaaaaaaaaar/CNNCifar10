import numpy as np
import matplotlib.pyplot as plt
from classes.Model import Model
from preprocess import test_images, class_names
from random import randint

weights_path = "weights/weights.h5"
base_model = Model()
model = base_model.create_model()
model.load_weights(weights_path)

i = randint(0, 9999)
predictions = model.predict(test_images)
prediction = np.argmax(predictions[i])
print(f"{class_names[prediction]} (%{predictions[i][prediction]:.2f})")
plt.figure()
plt.imshow(test_images[i], cmap=plt.cm.binary)
plt.title(f"{class_names[prediction]} (%{predictions[i][prediction]:.2f})")
plt.show()
