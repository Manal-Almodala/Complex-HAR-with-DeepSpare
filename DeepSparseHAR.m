

%% Construct Deep Network Using Autoencoders
% Load the sample data.
%%
clc;
clear;
diary('dataset_Results.txt'); % used to save the datasets
datetime;
M = load ('dataset.mat'); %dataset.mat represent each of the tri-xial accelerometer, magnitude vectors and rotational angle
K = struct2cell(M);
Y = cell2mat(K);
%clear XM;
NoLoops1=10;

%%
TAcc=zeros(2,1); %total accuracy
NoClasses = 13;
AllConfMat=zeros(NoClasses,NoClasses,NoLoops1);
AllAcc = zeros(NoLoops1,1);
%cnt=5;
%hiddenSize=20;
%for loop1 = 1:5    
%hiddenSize = hiddenSize+cnt;

tic
for Loop1 = 1:NoLoops1
Y = Y(randperm(size(Y,1)),:);

[r,c]  = size (Y);
Thrs = 0.7;

TrData  = Y(1:round(r*Thrs),1:c-1);
TsData  = Y(round(r*Thrs)+1:end,1:c-1);
TrLbls  = Y(1:round(r*Thrs),end);
TsLbls  = Y(round(r*Thrs)+1:end,end);

%%
%Transforming the labels for autoencoder and confusion matrix in binary
%forms
TrL = BinaryLbls(TrLbls);
TsL = BinaryLbls(TsLbls);
%
%% 

%[X,T] = wine_dataset;
X= TrData';
T=TrL';
%%
% Train an autoencoder with a hidden layer of size 10 and a linear transfer 
% function for the decoder. Set the L2 weight regularizer to 0.001, sparsity regularizer 
% to 4 and sparsity proportion to 0.05.
%%
hiddenSize=60;
autoenc1 = trainAutoencoder(X,hiddenSize,...
    'MaxEpoch',500,...
    'UseGPU',false,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',1.5,...
    'SparsityProportion',0.05,...
    'EncoderTransferFunction','logsig',...
    'DecoderTransferFunction','logsig');
%% 
% Extract the features in the hidden layer.
%%
features1 = encode(autoenc1,X); %feature extraction process
%% 
% Train a second autoencoder using the features from the first autoencoder. 
% Do not scale the data.
%%
hiddenSize1 = 40; %set the various parameter of the autoencoder algorithm using the matlab documentation
%hiddenSize = loop1-2;  %27
autoenc2 = trainAutoencoder(features1,hiddenSize1,...
    'MaxEpoch',500,...
    'UseGPU',false,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',1.5,...
    'SparsityProportion',0.05,...
    'EncoderTransferFunction','logsig',...
    'DecoderTransferFunction','logsig',...
    'ScaleData',false);
%%
features2 = encode(autoenc2,features1);
%%
hiddenSize2 = 30;
%hiddenSize = loop1-2;  %27
autoenc3 = trainAutoencoder(features2,hiddenSize2,...
    'MaxEpoch',500,...
    'UseGPU',false,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',1.5,...
    'SparsityProportion',0.05,...
    'EncoderTransferFunction','logsig',...
    'DecoderTransferFunction','logsig',...
    'ScaleData',false);
%
features3 = encode(autoenc3,features2);

%% 
% Train a softmax layer for classification using the features, |features2|, 
% from the second autoencoder, |autoenc2|.
%%
softnet = trainSoftmaxLayer(features3,T,'LossFunction','crossentropy', 'MaxEpoch',500);
%% 
% Stack the encoders and the softmax layer to form a deep network.
%%
deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);
%% 
% Train the deep network on the wine data.
%%
deepnet = train(deepnet,X,T);
%view(deepnet)
%% 
% Estimate the wine types using the deep network, |deepnet|.
%%
predictedValues = deepnet(TsData');
%% 
%clc
A = predictedValues';
[R,C] = size(A);
TsL_1C=zeros(R,1);
for Loop2 = 1:R
    [~,idx] = find(A(Loop2,:)== max(A(Loop2,:)));
    TsL_1C(Loop2,1)=idx;
end
%%
%h = plotconfusion(TsL',predictedValues);
%%
confMatAE = confusionmat(TsLbls,TsL_1C); %Compute the accuracy using confusion matrix
AllConfMat(:,:,Loop1) = confMatAE;
%[Result,RefereceResult]=confusion.getValues(AllConfMat(:,:,Loop1))
confMatAE = confMatAE./sum(confMatAE,2);
Acc = mean(diag(confMatAE))*100;  
AllAcc(Loop1,1)=Acc;
save('XTMR_Combined_Conf.mat','AllConfMat');
%% 
end
toc
%All_Acc=AllAcc
AllAcc;
Avg_Acc = mean(AllAcc);
Std = std(AllAcc);
%end
test
diary off;

%%
function [TsL] = BinaryLbls(Lbls) %setting the label values for training and testing data
[M,~] = size(Lbls);
TsL=zeros(M,13);
 for I = 1:M
    if Lbls(I,1) == 1
        TsL(I,1)=1;
    elseif Lbls(I,1) == 2
        TsL(I,2)=1;
    elseif Lbls(I,1) == 3
        TsL(I,3)=1;
    elseif Lbls(I,1) == 4
        TsL(I,4)=1;
    elseif Lbls(I,1) == 5
        TsL(I,5)=1;
    elseif Lbls(I,1) == 6
        TsL(I,6)=1;
    elseif Lbls(I,1) == 7
        TsL(I,7)=1;
    elseif Lbls(I,1) == 8
        TsL(I,8)=1;
    elseif Lbls(I,1) == 9
        TsL(I,9)=1;
    elseif Lbls(I,1) == 10
        TsL(I,10)=1;
    elseif Lbls(I,1) == 11
        TsL(I,11)=1;
    elseif Lbls(I,1) == 12
        TsL(I,12)=1;
    elseif Lbls(I,1) == 13
        TsL(I,13)=1;
    end   
end
end