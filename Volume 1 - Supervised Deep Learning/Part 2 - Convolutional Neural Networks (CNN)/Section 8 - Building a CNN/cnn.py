# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN

classifier = Sequential()

# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# Step 2 - Max Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))
