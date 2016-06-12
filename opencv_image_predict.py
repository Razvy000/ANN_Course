import cv2
import numpy as np

from ann import ANN
import ann_util

# load the image
digit = cv2.imread("./data/mnist/3.png", cv2.IMREAD_GRAYSCALE)
#digit = cv2.imread("./data/mnist/7.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("original digit", digit)

# resize it, it needs to be 28 x 28 for our MNIST ANN
small = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_NEAREST)
cv2.imshow("smaller digit", small)

# scale the values to 0..1
small -= small.min()
small /= small.max()

# from a matrix of 28x28 make it a vector of 768x1
print small.shape
small = small.reshape((784, 1))
print small.shape

# load the ANN
nn = ann_util.deserialize('models/nn_mnist_iter800000.pickle')

# predict output vector 
prediction = nn.predict(small)
print prediction

# the correct label is the index at which we have the largest values
label = np.argmax(prediction)
print "The predicted label is:",  label

# blocking wait (press any key)
k = cv2.waitKey(0)