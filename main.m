% implement of NMF Net
%% FCN/ACN1/ACN2
clear;clc;
%net = fcn;
net = 'acn1';
%net = 'acn2';
training = 1;
%% convexnmf/seminmf/nmf
%method = 'convex';
method = 'semi';
%method = 'nmf';
%% setting
batch = 2000;
if net == 'acn1'
    layer = 4;
    filter_size = [5,5,3,1];
    stride = [2,2,2,1];
    channel = [1,16,16,16,10];
elseif net == 'acn2'
    layer = 6;
    filter_size = [5,3,5,3,3,1];
    stride = [1,2,1,2,1,1];
    channel = [1,16,16,16,16,16,10];
end
r = channel;
n = filter_size;
%% data & preprocessing
[TrainImages,TestImages,TrainLabels,TestLabels] = load_dataset('MNIST');
if training == 1
for k = 15:21
        Input = cell(1,layer);
        Input{1} = TrainImages(:,1+batch*(k-1):batch*(k));
        W = cell(1,layer);
        H = cell(1,layer);
        Size_input = compute_size(28,stride,filter_size);
        for i = 1:layer
            Input{i} = process_nmf(Input{i},filter_size(i),stride(i),channel(i), Size_input(i:i+1),batch);
            fprintf('Layer %d NMF begins',i);
            if method == 'semi'
                [A2,Y2,~,t(i),error(i)]=seminmfrule(Input{i},r(i+1));
                W{i} = A2;
                H{i} = Y2;
            elseif method == 'convex'
                [A,Y,~,t(i),error(i)]=convexnmfrule(Input{i},r(i+1));
                W{i} = TestImages*A;
                H{i} = Y;
            elseif method == 'nmf'
                [A3,Y3,~,t(i),error(i)]=nmfrule(Input{i},r(i+1));
                W{i} = A3;
                H{i} = Y3;
            end
            %% ReLU
            temp = pinv(W{i})*Input{i};
            temp(temp<0) = 0;
            Input{i+1} = temp;
        end
        filename = ['nmf_net_', num2str(k)];
        save(filename,'W','H','error','t','Input');
    end
elseif training == 0
    %% process
    WW = cell(1,layer);
    HH = cell(1,layer);
    for k = 1:floor(size(TrainImages,2)/batch)
        filename = ['nmf_net_', num2str(k)];
        load(filename);
        for i = 1:length(WW)
            WW{i} = WW{i}+W{i};
            HH{i} = [HH{i} H{i}];
        end
    end
    for i = 1:length(WW)
        WW{i} = WW{i}/floor(size(TrainImages,2)/batch)
    end
    save('nmf_net');
    %% inference
    for i = 1:length(WW)
        temp = pinv(WW{i})*Input{i};
        temp(temp<0) = 0;
        Input{i+1} = temp;
        if i < length(WW)
            Input{i+1} = process_nmf(Input{i+1},filter_size(i),stride(i),channel(i), Size_input(i:i+1),batch);
        end
    end
    model = svmtrain(TrainLabels(1:batch), Input{layer+1}' ,'libsvm_options');
end

function Size_input = compute_size(image_size,stride,filter_size)
Size_input = [image_size, floor((image_size-(filter_size(1)-stride(1)))/stride(1))];
layer = length(stride);
for i = 2:layer
    temp = floor((Size_input(end)-(filter_size(i)-stride(i)))/stride(i));
    Size_input = [Size_input temp];
end
end

function Pro_image = process_nmf(input,filter_size,stride,channel, Size_input,batch)
% input is [Size_input^2 channel batch]
temp_testimage = reshape(input,[Size_input(1),Size_input(1),channel*batch]);
%k = 1;
for i = 1:Size_input(2)
    for j = 1:Size_input(2)
        Pro_image((i-1)*filter_size+1:i*filter_size,(j-1)*filter_size+1:j*filter_size,:) = temp_testimage(1+(i-1)*stride:filter_size+(i-1)*stride,(j-1)*stride+1:filter_size+(j-1)*stride,:);
        %Pro_image(:,:,k) = temp_testimage(1+(i-1)*stride:filter_size+(i-1)*stride,(j-1)*stride+1:filter_size+(j-1)*stride,:);
        %k = k+1;
    end
end
Pro_image = reshape(Pro_image,[filter_size,Size_input(2),filter_size,Size_input(2),channel,batch]);
Pro_image = permute(Pro_image,[1,3,5,2,4,6]);
Pro_image = reshape(Pro_image,[filter_size^2*channel,Size_input(2)^2*batch]);
Pro_image = [ones(1,size(Pro_image,2));Pro_image];
end
