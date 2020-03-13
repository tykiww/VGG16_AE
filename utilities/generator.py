import cv2
import numpy as np
from random import randint
from keras.utils import Sequence
from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv2D


def speckle(X_imgs):
    X_imgs_copy = [X_imgs.copy()]
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy[0]



def poisson(image):
    poisson_noise = image.copy() 
    image = image + np.random.poisson(poisson_noise*.6)   # .6 is an artificial linear noise factor
    return np.clip(image,0,255)

def gaussian(image):
    gaussian_noise = image.copy()
    cv2.randn(gaussian_noise, 0, 150)
    image = image + gaussian_noise
    return image

def scaling(image):
    row, col, ch = image.shape

    check = randint(1, 3)
    scale = randint(1, 9)
    scale2 = randint(1, 9)
    if check == 1:
        image = cv2.resize(image, None, fx=(scale), fy=(scale2), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (row, col))
        return image

    elif check == 2:
        image = cv2.resize(image, None, fx=(scale), fy=(scale2), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (row, col))
        return image

    elif check == 3:
        image = cv2.resize(image, None, fx=(scale), fy=(scale2), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (row, col))
        return image

    else:
        image = cv2.resize(image, None, fx=(scale), fy=(scale2), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (row, col))
        return image

def rotation(image):
    rows, cols, ch = image.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), randint(0, 360), 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def translation(image):
    rows, cols, ch = image.shape

    m = np.float32([[1, 0, int(randint(1, rows) * .45)], [0, 1, int(randint(1, cols) * .45)]])
    image = cv2.warpAffine(image, m, (cols, rows))
    return image

def affine(image):
    rows, cols, ch = image.shape

    pts1 = np.float32([[50, randint(2, 15)], [200, randint(10, 50)], [50, randint(200, 250)]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def edgedetection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges


def color(image):
    image = (255 - image)
    return image


def inverse(image):
    image = np.int16(image)

    contrast = randint(1, 60)
    brightness = randint(1, 60)

    img = image * (contrast / 127 + 1) - contrast + brightness

    img = np.clip(img, 0, 255)
    image = np.uint8(img)
    return image

def grayscale(image):
  image = np.mean(image,-1)
  return image

# Create Function random change
def random_change(image):
    x = randint(1, 6)
    if x == 1:
        return image
    if x == 2:
        return scaling(image)
    if x == 3:
        return gaussian(image)
    if x == 4:
        return poisson(image)
    if x == 5:
        return speckle(image)
    if x == 6:
        return inverse(image)

# Create class data generator
class data_generator(Sequence):

    def __init__(self, data, batch_size=128, noisy=False, shuffle=True):

        self.data = data
        self.data_noisy = None
        self.index = [i for i in range(self.data.shape[0])]

        self.batch_size = batch_size
        self.noisy = noisy
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, index):

        if self.noisy:
            input, target = [], []
            for i in range(index, index + self.batch_size):
                input.append(random_change(self.data[self.index[i]]))
                target.append(self.data[self.index[i]])
            input = np.array(input)/255.0
            input = np.reshape(input, (len(input), 256, 256, 3))
            target = np.array(target)/255.0
            target = np.reshape(target, (len(target), 256, 256, 3))
            return input,target
        else:
            input, target = [], []
            for i in range(index, index + self.batch_size):
                input.append(self.data[self.index[i]])
                target.append(self.data[self.index[i]])
            input = np.array(input)/255.0
            input = np.reshape(input, (len(input), 256, 256, 3))
            target = np.array(target)/255.0
            target = np.reshape(target, (len(target), 256, 256, 3))
            return input,target

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index)
