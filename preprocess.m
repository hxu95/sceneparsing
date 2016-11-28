% run through images, resize and modify -> save intermediate results 
% preprocessing for prediction
close all; clc; clear;

% initialize the network
% net = caffe.Net(model_definition, model_weights, 'test');

% path to image(.jpg) and annotation(.png) and generated prediction(.png)
pathImg = fullfile('/home/hxu/sceneparsing/sampleData', 'images');
process_folder = 'intermediate';
pathProcess = fullfile('/home/hxu/sceneparsing/sampleData', process_folder);

if (~exist(pathProcess, 'dir'))
	mkdir(pathProcess);
end

filesImg = dir(fullfile(pathImg, '*.jpg'));
for i = 1: numel(filesImg)
    % read image
    fileImg = fullfile(pathImg, filesImg(i).name);
    fileTemp = fullfile(pathProcess, strrep(filesImg(i).name, '.jpg', '.png'));
    
    im = imread(fileImg);
  	
    % resize image to fit model description
    im_inp = double(imresize(im, [384,384])); 

    % change RGB to BGR
    im_inp = im_inp(:,:,end:-1:1);

    % substract mean and transpose
    im_inp = cat(3, im_inp(:,:,1)-109.5388, im_inp(:,:,2)-118.6897, im_inp(:,:,3)-124.6901);
    im_inp = permute(im_inp, [2,1,3]);
    
    imwrite(im_inp, fileTemp);    
end