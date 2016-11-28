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
#manual path definition
#sys.path.append('/home/hxu/caffe/python')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import caffe
import os

base_dir = 'home/hxu/sceneparsing'
models_dir = os.path.join(base_dir, 'models')
print models_dir
#sys.path.append('/home/hxu/caffe/python')

# define model and weights
model_type = 'FCN' # Dilated'
if (model_type is 'FCN'):
  model_definition = 'deploy_FCN.prototxt'
  model_weights = os.path.join(models_dir, 'FCN_iter_160000.caffemodel')
elif (model_type is 'Dilated'):
  model_definition = os.path.join(models_dir, 'deploy_DilatedNet.prototxt')
  model_weights = os.path.join(models_dir, 'DilatedNet_iter_120000.caffemodel')

# output folder
prediction_folder = os.path.join(base_dir, 'sampleData/predictions_' + model_type)
print prediction_folder

if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)

#load the net
net = caffe.Net(model_definition, model_weights, caffe.TEST);

# load input images that we want to do segmentation on
inputs = os.path.join(base_dir, 'sampleData/intermediate')
print inputs

# loop through input images
for filename in os.listdir(inputs):
  print filename
  img = cv2.imread(os.path.join(inputs, filename))
  out = net.forward(img)
  outfile = os.path.join(prediction_folder, filename)
  cv2.imwrite(outfile, out)
