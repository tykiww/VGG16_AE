

from utilities.models import VGGAE, latest_file, set_pointer
from utilities.generator import datagenerator
from utilities.data import retrieve_images
from keras.callbacks import ReduceLROnPlateau


def main(inp_shape, num_epochs, batch_n, model_file,
         opt, lr, loss_metric, display_summary):
  
  # Retrieve Images
  train, tests = retrieve_images()
  # Retrieve weights
  lf = latest_file(pathname = model_file)
  # Retrieve Model
  vggae = VGGAE(inp_shape)
  
  # Setup Model
  autoencoder.compile(optimizer=opt, loss=loss_metric)
  
  if display_summary:
    autoencoder.summary()
  else:
    pass
  
  if lf == 0:
    pass:
  else:
    autoencoder.load_weights(lf)
  
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=3, min_lr=lr)

  # Train Auto Encoder
  autoencoder.fit_generator(datagenerator(data = train, batch_size=batch_n, noisy = True),
                            steps_per_epoch=  int(len(train)/batch_n),
                            epochs=num_epochs,
                            verbose=1,
                            callbacks = [reduce_lr,
                                         set_pointer(origin=model_file)
                                         ],
                            validation_data=datagenerator(data = tests, batch_size=int(batch_n/2), noisy = False),
                            validation_steps=1) # .10692 is the threshold for a clear image.
    
    
if __name__ == "__main__":
  main(inp_shape = (256,256,3),
       num_epochs = 200,
       batch_n = 32,
       model_file = 'logging/check/*',
       opt = 'adadelta',
       lr = 0.0001,
       loss_metric = 'binary_crossentropy',
       display_summary = False)


