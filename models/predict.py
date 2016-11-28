'''
 select the pre-trained model. Use 'FCN' for 
% the Fully Convolutional Network or 'Dilated' for DilatedNet
% You can download the FCN model at 
% http://sceneparsing.csail.mit.edu/model/FCN_iter_160000.caffemodel
% and the DilatedNet model at
% http://sceneparsing.csail.mit.edu/model/DilatedNet_iter_120000.caffemodel
'''

# Set python path here
import sys
from PIL import Image
#manual path definition
sys.path.append('/home/hxu/caffe_main/python')

import matplotlib.pyplot as plt
import cv2
import numpy as np
import caffe
import Image
import os

base_dir = '/home/hxu/sceneparsing'
models_dir = os.path.join(base_dir, 'models')
#sys.path.append('/home/hxu/caffe/python')

# define model and weights
model_type = 'FCN' # Dilated'
if (model_type is 'FCN'):
  model_definition = 'deploy_FCN.prototxt'
  model_weights = os.path.join(models_dir, 'FCN_iter_160000.caffemodel')
elif (model_type is 'Dilated'):
  model_definition = 'deploy_DilatedNet.prototxt'
  model_weights = os.path.join(models_dir, 'DilatedNet_iter_120000.caffemodel')

# output folder
prediction_folder = os.path.join(base_dir, 'sampleData/predictions_' + model_type)

if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)

#load the net
net = caffe.Net(model_definition, model_weights, caffe.TEST)
# net = caffe.Net('conv.prototxt', caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# load input images that we want to do segmentation on
inputs = os.path.join(base_dir, 'sampleData/intermediate')
print inputs

# loop through input images
for filename in os.listdir(inputs):
  print filename
  # img = cv2.imread()
  
  # img = img.astype(np.float32)/255.
  # img = img[...,::-1]
  path = os.path.join(inputs, filename)
  # img = cv2.imread(path)
  img = caffe.io.load_image(path)
  '''
  # im = np.array(Image.open(os.path.join(inputs, filename)))
  im_input = img[np.newaxis, np.newaxis, :, :]
  net.blobs['data'].reshape(*im_input.shape)
  # net.blobs['data'].data[...] = transformer.preprocess('data', img)
  # out = net.forward()
  net.blobs['data'].data[...] = im_input
  # out = net.forward([img])
  '''

  # http://stackoverflow.com/questions/29124840/prediction-in-caffe-exception-input-blob-arguments-do-not-match-net-inputs
  #net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path))

  # Pad an image so that it has a shape of a multiple of 32
  # Reshape and fill input with the image
  net.blobs['data'].reshape(1, 3, *img.shape[:2])
  net.blobs['data'].data[0, ...] = np.rollaxis(img, 2)
  # Predict and get the outputs albedo and shading
  out = net.forward()

  print out

  a = out['upscore8'].copy()

  outfile = os.path.join(prediction_folder, filename)
  plt.figure()
  plt.imshow(a)
  plt.show()
  # cv2.imwrite(outfile, out)
  plt.imsave(outfile, a)
