function [cost] = ann_cost(x)
% ANN Cost Function for TLBO_PSO_GA Optimizations
% x = [Number of Hidden Layers, hiddenLayerSize, Learning Rate]
 
% Load input data (factors)
factors = csvread('D:\PRADEEP\ANN_hyperparameter optimization\INPUT\factors.csv');
% Load target data (responses)
responses = csvread('D:\PRADEEP\ANN_hyperparameter optimization\INPUT\sresponses.csv');
 
% Set decision variables
hiddenLayers = round(x(1));        % Number of Hidden Layers
hiddenLayerSize = round(x(2));     % hiddenLayerSize
learningRate = x(3);               % Learning Rate
 
% Construct the ANN model
x = factors';
t = responses';
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
net = fitnet(hiddenLayerSize*ones(1,hiddenLayers),trainFcn);
 % Specify hidden layer structure
net.trainParam.lr = learningRate; % Set learning rate
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand';  
net.divideMode = 'sample';  
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'mse';
net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression', 'plotfit'};
 
% Train the ANN model
[net,tr] = train(net,x,t);
 
% Evaluate the ANN model's performance using MSE
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
cost = performance;
assignin('base', 'predicted_values', y);
end
