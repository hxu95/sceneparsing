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
#sys.path.append('/home/hxu/caffe_main/python')

import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.misc
import caffe
import Image
import os

base_dir = '/afs/csail.mit.edu/u/y/ylkuo/project/cv_final/sceneparsing/'
models_dir = os.path.join(base_dir, 'models')
#sys.path.append('/home/hxu/caffe/python')

# define model and weights
model_type = 'FCN' # Dilated'
if (model_type is 'FCN'):
  model_definition = 'test_FCN.prototxt'
  #model_weights = os.path.join(models_dir, 'FCN_iter_160000.caffemodel')
  model_weights = 'FCN_iter_160000.caffemodel'
elif (model_type is 'Dilated'):
  model_definition = 'deploy_DilatedNet.prototxt'
  #model_weights = os.path.join(models_dir, 'DilatedNet_iter_120000.caffemodel')
  model_weights = 'DilatedNet_iter_120000.caffemodel'

# output folder
prediction_folder = os.path.join(base_dir, 'sampleData/predictions_' + model_type)

if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)

#load the net
net = caffe.Net(model_definition, model_weights, caffe.TEST)
# net = caffe.Net('conv.prototxt', caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# load input images that we want to do segmentation on
#inputs = os.path.join(base_dir, 'sampleData/intermediate')
inputs = os.path.join(base_dir, 'sampleData/images')

# loop through input images
for filename in os.listdir(inputs):
  print filename
  path = os.path.join(inputs, filename)
  img = caffe.io.load_image(path)
  img = img[..., ::-1]
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

  # Pad an image so that it has a shape of a multiple of 32
  # Reshape and fill input with the image
  net.blobs['data'].reshape(1, 3, *img.shape[:2])
  net.blobs['data'].data[0, ...] = np.rollaxis(img, 2)
  # Predict and get the outputs albedo and shading
  out = net.forward()

  if (model_type is 'FCN'):
    a = out['upscore8'].copy()
  elif (model_type is 'Dilated'):
    a = out['fc_final_up'].copy()
  a = a[0]
  out_a = np.argmax(a, axis=0)
  #out_a = np.transpose(out_a)
  out_a = out_a.astype(np.uint8) - 1
  print img.shape
  out_a = scipy.misc.imresize(out_a, (img.shape[0], img.shape[1])) 

  outfile = os.path.join(prediction_folder, filename)
  plt.figure()
  plt.imshow(out_a, cmap='gray')
  plt.show()
  plt.imsave(outfile, out_a, cmap='gray')
  #break
