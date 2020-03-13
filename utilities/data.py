import tables
from sklearn.model_selection import train_test_split
import gc


def retrieve_images(path = 'data/all_images.h5')
  h5_path = path
  hdf5_file = tables.open_file(h5_path, mode='r+')
  
  tot = np.shape(hdf5_file.root.images)
  train = hdf5_file.root.images[0:int(0.8*tot[0])]
  tests = hdf5_file.root.images[int(0.8*tot[0]):tot[0]]
  
  hdf5_file.close()
  gc.collect()
  return train, tests

