function [TrainImages,TestImages,TrainLabels,TestLabels] = load_dataset(Dataset)
if strcmpi(Dataset,'MNIST')
    clear all;clc
    TrainImages = loadMNISTImages('train-images-idx3-ubyte');
    TestImages = loadMNISTImages('t10k-images-idx3-ubyte');
    TrainLabels = loadMNISTLabels('train-labels-idx1-ubyte');
    TestLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
elseif strcmpi(Dataset,'SVHN')
    clear all;clc
    M = load('train_32x32.mat');
    N = load('test_32x32.mat');
    TrainImages = reshape(M.X,[32*32*3,size(M.X,4)]);
    TestImages = reshape(N.X,[32*32*3,size(N.X,4)]);
    TrainLabels = M.y-1;
    TestLabels = N.y-1;
    TrainImages = double(TrainImages);
    TestImages = double(TestImages);
elseif strcmpi(Dataset,'CIFAR')
    clear all;clc
    untar('cifar-10-matlab.tar.gz','CIFAR');
    load_data;
    for i = 1:5
        TrainImages(i*10000-9999:i*10000,:) = M(i).data;
        TrainLabels(i*10000-9999:i*10000,:) = M(i).labels;
    end
    TrainImages = double(permute(TrainImages,[2,1]))/255;
    TrainLabels = double((TrainLabels));
    TestImages = double(permute(N.data,[2,1]))/255;
    TestLabels = double((N.labels));
elseif strcmpi(Dataset,'MSRC')

end
end