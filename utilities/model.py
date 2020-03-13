

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Flatten,Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16



def VGGAE(input_shape):
  
  # Setup
  inp = Input(shape=input_shape)
  
  ################### encoder ###################
  encoded = VGG16(weights='imagenet', include_top = False)(inp)
  
  ################### latent ###################
  x = Conv2D(512, (3, 3), activation='relu', padding='same',name='latent')(encoded)
  x = UpSampling2D((2,2))(x)  
    
  ################### decoder ###################
        
  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2,2))(x)

  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2,2))(x)

  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2,2))(x)     
     
  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2,2))(x)        
 
  # Block 1
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
  
  return Model(inputs = inp, ouputs = decoded)





