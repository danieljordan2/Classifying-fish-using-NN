# USAGE
# python keras_salmon.py --output output/keras_mnist.png

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())


# construct the Salmon (freshwater and saltater) dataset
X = np.array([[83, 510], [86, 505], [94, 490], [118, 490], [86, 480], [98, 480], [101, 472], [120, 472], [90, 470], [100, 470], [101, 470], [105, 470], [75, 450], [83, 452], [85, 450], [85, 442], [75, 440], [93, 440], [105, 440], [52.5, 425], [78, 431], [82, 430], [95, 431], [105, 432], [87, 422], [95, 427], [102, 428], [114, 427], [109, 420], [111, 421], [126, 422], [95, 411], [70, 397], [80, 399], [84, 399], [87, 402], [92, 404], [98, 403], [98, 402], [104, 404], [121, 402], [106, 439], [109, 398], [112, 394], [114, 397], [107, 368], [118, 382], [126, 371], [136, 357], [95, 430], [135, 440], [129, 420], [156, 420], [128, 400], [144, 403], [152.5, 403], [178, 408], [129, 390], [140, 390], [149, 392], [152.5, 394], [154, 390], [128, 382], [134, 382], [148, 382], [152, 381], [170, 395], [120, 359], [133, 373], [138, 371], [140, 373], [148, 372], [163, 370], [170, 375], [123, 352], [140, 351], [162.5, 369], [90, 385], [115, 355], [117, 356], [135, 356], [145, 356], [150, 355], [152.5, 354], [155, 352], [123, 350], [125, 343], [126, 342], [131, 342], [144, 342], [107, 340], [116, 344], [124, 341], [144, 339], [150, 340], [112.5, 330], [114, 323], [122, 304], [152, 301], [118,381]])

y = np.array([[1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

# scale data to the range of [0, 1]
Xmax = np.max(X)
Xmin = np.min(X)
Xdelta = Xmax - Xmin
X = (X - Xmin) / Xdelta

# define the 2-2 architecture using Keras
model = Sequential()
model.add(Dense(2, input_shape=(2,), activation="sigmoid"))
model.add(Dense(2, activation="softmax"))

# train the model usign SGD
print("[INFO] training network...")
sgd = SGD(0.0001)
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
H = model.fit(X, y, validation_data=(X, y), epochs=80, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(X, batch_size=32)
print(classification_report(y.argmax(axis=1), predictions.argmax(axis=1), target_names=None))
print(predictions)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 80), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 80), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 80), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 80), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
