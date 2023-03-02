%Close all open figures
close all
%Clear the workspace
clear
%Clear the command window
clc
%%


%loading the data
imds1 = imageDatastore(fullfile('C:\Users\maria\Downloads\archive (2)'),...
"IncludeSubfolders",true, "LabelSource","foldernames");

%returns a summary table of the labels in imds and the number of files associated with each.
labelCount = countEachLabel(imds1);
TotalImages = sum(labelCount.Count)


figure;
perm = randperm(TotalImages,20);
for i = 1:20
subplot(4,5,i);
imshow(imds1.Files{perm(i)});
end


%reads the Ith image file from the datastore imds and returns the image data img
img = readimage(imds1,1);
size(img);

inputSize = [100 100];
imds1.ReadFcn = @(loc)imresize(imread(loc),inputSize);
imshow(preview(imds1));

%Resizing the image
%A = imread(imds1);
%imds = imresize(imds1,[125 300],antialiasing = False);

%numTrainFiles=800;
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds1,0.8,0.1,'randomize');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20]);

 A1 =augmentedImageDatastore([100 100],imdsTrain,"ColorPreprocessing","rgb2gray",'DataAugmentation',imageAugmenter);
 v1 =augmentedImageDatastore([100 100],imdsValidation,"ColorPreprocessing","rgb2gray",'DataAugmentation',imageAugmenter);
 T1 = augmentedImageDatastore([100 100],imdsTest,"ColorPreprocessing","rgb2gray");

layers = [imageInputLayer([100 100 1]) ...
    convolution2dLayer(3,8,'Padding','same') ...
    batchNormalizationLayer reluLayer maxPooling2dLayer(2,'Stride',2) ...
    convolution2dLayer(3,16,'Padding','same') ...
    batchNormalizationLayer reluLayer maxPooling2dLayer(2,'Stride',2) ...
    convolution2dLayer(3,32,'Padding','same') ...
    batchNormalizationLayer reluLayer maxPooling2dLayer(2,'Stride',2) ...
    maxPooling2dLayer(2,'Stride',2) convolution2dLayer(3,64,'Padding','same') ...
    batchNormalizationLayer  ...
    reluLayer  ...
    fullyConnectedLayer(156) ...
    softmaxLayer classificationLayer];

options = trainingOptions('sgdm', ...
 'InitialLearnRate',0.01, ...
 'MaxEpochs',1, ...
 'Shuffle','every-epoch', ...
 'ValidationData',v1, ...
 'ValidationFrequency',30, ...
 'Verbose',false, ...
 'Plots','training-progress');

net = trainNetwork(A1,layers,options);
%%

YPred = classify(net,T1);
YValidation = imdsTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);

%%confMat = plotconfusion(YPred,YValidation);


Y = randi(131,103,1);
Y_hat = Y;
Y_hat(randi(103,130,1)) = randi(131,130,1);
[c,order] = confusionmat(Y,Y_hat);
plotconfusion(Y,Y_hat);