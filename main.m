% implement of NMF Net
%% FCN/ACN1/ACN2
clear;clc;
net = 'acn1';% 'fcn' 'acn2'
training = 1; % 1 nmf|| 0 Inference svm || 2 svm acc|| -1 process || 4 Label info||3 layer svm
%% convexnmf/seminmf/nmf
method = 'semi'; %'convex' ||'nmf'
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
Size_input = compute_size(28,stride,filter_size);
%% data & preprocessing
[TrainImages,TestImages,TrainLabels,TestLabels] = load_dataset('MNIST');
if training == 1
    for k = 1:floor(size(TrainImages,2)/batch)
        Input = cell(1,layer);
        Input{1} = TrainImages(:,1+batch*(k-1):batch*(k));
        W = cell(1,layer);
        H = cell(1,layer);
        for i = 1:layer
            Input{i} = process_nmf(Input{i},filter_size(i),stride(i),channel(i),Size_input(i:i+1),batch);
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
elseif training == -1
    %% process, combine different batch
    WW = cell(1,layer);
    HH = cell(1,layer);
    IInput = [];
    for k = 1:floor(size(TrainImages,2)/batch)
        filename = ['nmf_net_', num2str(k)];
        load(filename);
        for i = 1:length(WW)
            if k == 1
                WW{i} = [WW{i} W{i}];
            elseif k > 1
                WW{i} = WW{i}+W{i};
            end
            HH{i} = [HH{i} H{i}];
        end
        IInput = [IInput Input{1}];
    end
    for i = 1:length(WW)
        WW{i} = WW{i}/floor(size(TrainImages,2)/batch)
    end
    clear Input;
    Input{1} = IInput;
    clear IInput;
    save('nmf_net');
elseif training == 0
    %% inference
    load('nmf_net');
    for i = 1:layer
        H_size(i) = size(H{i},2);
    end
    %% training set
    %     for i = 1:length(WW)
    %         temp = pinv(WW{i})*Input{i};
    %         temp(temp<0) = 0;
    %         Input{i+1} = temp;
    %         if i < length(WW)
    %             IIInput{i+1} = [];
    %             for k = 1:floor(size(TrainImages,2)/batch)
    %                  temp1 = process_nmf(Input{i+1}(:,(k-1)*H_size(i)+1:(k)*H_size(i)),filter_size(i+1),stride(i+1),channel(i+1), Size_input(i+1:i+2),batch);
    %                  IIInput{i+1} = [IIInput{i+1} temp1];
    %             end
    %         elseif i == length(WW)
    %             IIInput{i+1} = Input{i+1};
    %         end
    %         Input{i+1} = IIInput{i+1};
    %     end
    %     % single
    %     for i = 1:layer
    %         Input{i} = single(Input{i});
    %     end
    %% testing set
    Input{1} = TestImages;
    tm = [];
    for i = 1:length(WW)
        temp11 = process_nmf(reshape(Input{1},[size(Input{1},1),H_size(1),batch]),filter_size(1),stride(1),channel(1), Size_input(1:2),batch);
        tm = [tm temp11];
        Input{1} = tm;
        
        temp = pinv(WW{i})*Input{i};
        temp(temp<0) = 0;
        Input{i+1} = temp;
        if i < length(WW)
            IIInput{i+1} = [];
            for k = 1:floor(size(TestImages,2)/batch)
                temp1 = process_nmf(Input{i+1}(:,(k-1)*H_size(i)+1:(k)*H_size(i)),filter_size(i+1),stride(i+1),channel(i+1), Size_input(i+1:i+2),batch);
                IIInput{i+1} = [IIInput{i+1} temp1];
            end
        elseif i == length(WW)
            IIInput{i+1} = Input{i+1};
        end
        Input{i+1} = IIInput{i+1};
    end
    % single
    for i = 1:layer
        Input{i} = single(Input{i});
    end
    save('nmf_final_test.mat','Input','WW','HH','-v7.3')
elseif training==2
    load('nmf_final_train.mat')
    num = 60000;
    temp_train = Input{layer+1}';
    temp_train = normalize(temp_train,1,'range');% scaling
    model_train = svmtrain(TrainLabels(1:num), temp_train(1:num,:),'-s 0 -t 2 -c 1 -g 0.07');
    %% test transfer
    load('nmf_final_test.mat')
    for i = 1:layer
        Input{i} = double(Input{i});
    end
    
    temp_test = Input{layer+1}';
    temp_test = normalize(temp_test,1,'range');
    model_test = svmtrain(TestLabels, temp_test,'-s 0 -t 2 -c 1 -g 0.07');
    [predict_label, accuracy, dec_values] = svmpredict(TrainLabels(1:num),temp_train(1:num,:) , model_train);
    [predict_label, accuracy, dec_values] = svmpredict(TestLabels,temp_test , model_train);
    [predict_label, accuracy, dec_values] = svmpredict(TrainLabels(1:num),temp_train(1:num,:) , model_test);
    [predict_label, accuracy, dec_values] = svmpredict(TestLabels,temp_test , model_test);
elseif training==3
    load('nmf_final_train.mat')
    for i = 1:4
        Input{i} = reshape(Input{i},[],60000);
    end
    
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
% input is [channel Size_input^2  batch]
temp_testimage = reshape(input,[channel, Size_input(1),Size_input(1),batch]);
temp_testimage = permute(temp_testimage,[2,3,1,4]);
temp_testimage = reshape(temp_testimage,[Size_input(1),Size_input(1),channel*batch]);
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