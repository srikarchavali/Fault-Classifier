clc;
close all;

%% Load trainig data
data = load('TrainingData');
datatest = load('TestData');
TrainingData = data.TrainingData;
TrainingData = table2array(TrainingData); %converting tabular data to array for eeasy callng
TestData = datatest.TestData;
TestData = table2array(TestData);

% Obtain number of rows and colums in test data
[n_rows,n_cols] = size(TrainingData);

%% Process data
% Shuffling training data
shuffle_idx = randperm(n_rows);
TrainingData = TrainingData(shuffle_idx, :);

% Splitting training data into training data and test data
XTrain0 = TrainingData(1:0.7*n_rows,1:18); %
YTrain = TrainingData(1:0.7*n_rows,end);
XTest0 = TrainingData(0.7*n_rows+1:end,1:18);
YTest = TrainingData(0.7*n_rows+1:end,end);

% obtain the number of samples/features and dimensionality of test data
[n_samples, n_dims] = size(XTrain0);

%% Evaluating model before any dimensionality reduction or feature transformation

% Evaluating using KNN
[accuracyknn,precisionknn,recallknn,F_measureknn] = knncperf(XTrain0,...
    YTrain, XTest0, YTest, n_dims);
disp("Accuracy of KNN model (no feature transformation or reduction): "...
    + string(accuracyknn));
disp("Precision = "  + string(precisionknn));
disp("recall = " + string(recallknn));
disp("F-measure = " + string(F_measureknn));

% Evaluating using decision tree
[accuracytree,precisiontree,recalltree,F_measuretree] = treecperf(XTrain0,...
    YTrain, XTest0, YTest, n_dims);
disp("Accuracy of decision tree model (no feature transformation or reduction): "...
    + string(accuracytree));
disp("Precision = "  + string(precisiontree));
disp("recall = " + string(recalltree));
disp("F-measure = " + string(F_measuretree));

% Evaluating using Naive Bayes
[accuracynb,precisionnb,recallnb,F_measurenb] = nbcperf(XTrain0, YTrain,...
    XTest0, YTest, n_dims);
disp("Accuracy of Naive bayes model (no feature transformation or reduction): "...
    + string(accuracynb));
disp("Precesion = "  + string(precisionnb));
disp("recall = " + string(recallnb));
disp("F-measure = " + string(F_measurenb));

%% Evaluate model after standardizing training data
%Data standardization
XTrainstd = zscore(XTrain0,[ ],1);
XTeststd = zscore(XTest0,[ ],1);

% Evaluating using KNN
[accuracyknn,precisionknn,recallknn,F_measureknn] = knncperf(XTrainstd,...
    YTrain, XTeststd, YTest, n_dims);
disp("Accuracy of KNN model (scaled data): " + string(accuracyknn));
disp("Precesion = "  + string(precisionknn));
disp("recall = " + string(recallknn));
disp("F-measure = " + string(F_measureknn));

% Evaluating using decision tree
[accuracytree,precisiontree,recalltree,F_measuretree] = treecperf(XTrainstd,...
    YTrain, XTeststd, YTest, n_dims);
disp("Accuracy of decision tree model (scaled data): " + string(accuracytree));
disp("Precesion = "  + string(precisiontree));
disp("recall = " + string(recalltree));
disp("F-measure = " + string(F_measuretree));

% Evaluating using Naive Bayes
[accuracynb,precisionnb,recallnb,F_measurenb] = nbcperf(XTrainstd,...
    YTrain, XTeststd, YTest, n_dims);
disp("Accuracy of Naive bayes model (scaled data): " + string(accuracynb));
disp("Precesion = "  + string(precisionnb));
disp("recall = " + string(recallnb));
disp("F-measure = " + string(F_measurenb));

%% Evaluate model with feature selection using MRMR

% feature selection using MRMR algorithm
[idx,scores] = fscmrmr(XTrain0, YTrain); %ranking features based on their significance 
figure('Name','MRMR Predictor ranking');
bar(scores(idx)); %Plotting the features on a bar graph
xlabel('Predictor rank');
ylabel('Predictor importance score');
xticklabels(idx);
n_dimsMRMR = 5; %selecting 5 most important features out of 18
XTrainMRMR = XTrain0(:, idx(1:n_dimsMRMR)); %Extracting first 5 important predictors
XTestMRMR = XTest0(:, idx(1:n_dimsMRMR));

% Evaluating with KNN
[accuracyknn,precisionknn,recallknn,F_measureknn] = knncperf(XTrainMRMR,...
    YTrain, XTestMRMR, YTest, n_dimsMRMR);
disp("Accuracy of KNN model (MRMR with "+string(n_dimsMRMR)+" features): " ...
    + string(accuracyknn));
disp("Precesion = "  + string(precisionknn));
disp("recall = " + string(recallknn));
disp("F-measure = " + string(F_measureknn));

% Evaluating with decision tree
[accuracytree,precisiontree,recalltree,F_measuretree] = treecperf(XTrainMRMR,...
    YTrain, XTestMRMR, YTest, n_dimsMRMR);
disp("Accuracy of decision tree model (MRMR with "+string(n_dimsMRMR)+" features): " ...
    + string(accuracytree));
disp("Precesion = "  + string(precisiontree));
disp("recall = " + string(recalltree));
disp("F-measure = " + string(F_measuretree));

% Evaluating using Naive Bayes
[accuracynb,precisionnb,recallnb,F_measurenb] = nbcperf(XTrainMRMR,...
    YTrain, XTestMRMR, YTest, n_dimsMRMR);
disp("Accuracy of Naive bayes model (MRMR with "+string(n_dimsMRMR)+" features): " ...
    + string(accuracynb));
disp("Precesion = "  + string(precisionnb));
disp("recall = " + string(recallnb));
disp("F-measure = " + string(F_measurenb));

%% Evaluate model with dimensionality reduction using PCA

%Dimensionality reduction using PCA
w = 1./var(XTrain0); % Feature scaling
[coeffs,score,latent,tsquared,explained] = pca(XTrain0,'VariableWeights',w);
XTrainPCA = (coeffs\XTrain0')';
XTestPCA = (coeffs\XTest0')';
figure('Name','PCA Scree Plot');
pareto(explained); %plotting the scree plot of the principal components
xlabel('Principal Component');
ylabel('Variance Explained (%)');
n_dimsPCA = 9;

% Evaluating with KNN
[accuracyknn,precisionknn,recallknn,F_measureknn] = knncperf(XTrainPCA,...
    YTrain, XTestPCA, YTest, n_dimsPCA);
disp("Accuracy of KNN model (PCA with "+string(n_dimsPCA)+...
" features): " + string(accuracyknn));
disp("Precesion = "  + string(precisionknn));
disp("recall = " + string(recallknn));
disp("F-measure = " + string(F_measureknn));


% Evaluating with tree model
[accuracytree,precisiontree,recalltree,F_measuretree] = treecperf(XTrainPCA,...
YTrain, XTestPCA, YTest, n_dimsPCA);
disp("Accuracy of decision tree model (PCA with "+string(n_dimsPCA)+...
" features): " + string(accuracytree));
disp("Precesion = "  + string(precisiontree));
disp("recall = " + string(recalltree));
disp("F-measure = " + string(F_measuretree));

% Evaluating using Naive Bayes
[accuracynb,precisionnb,recallnb,F_measurenb] = nbcperf(XTrainPCA,...
YTrain, XTestPCA, YTest, n_dimsPCA);
disp("Accuracy of Naive bayes model (PCA with "+string(n_dimsPCA)+...
" features): " + string(accuracynb));
disp("Precesion = "  + string(precisionnb));
disp("recall = " + string(recallnb));
disp("F-measure = " + string(F_measurenb));

%% Evaluate model with feature selection using NCA 

% Calculating feature weights using NCA 
sorted_dims_with_weights = sortrows([fscnca(XTrain0, YTrain).FeatureWeights, XTrain0'], 'descend')';
Weights = fscnca(XTrain0, YTrain).FeatureWeights;
figure('Name', 'NCA predictor weights');
plot(Weights,'ro'); % plotting features vs feature weights
n_dimsNCA = 4; % Selecting first 4 features having highest weight
XTrainNCA = sorted_dims_with_weights(2:end, 1:n_dimsNCA);
sorted_dims_with_weights = sortrows([fscnca(XTrain0, YTrain).FeatureWeights, XTest0'], 'descend')';
XTestNCA = sorted_dims_with_weights(2:end, :);

% Evaluating with KNN
[accuracyknn,precisionknn,recallknn,F_measureknn] = knncperf(XTrainNCA,...
    YTrain, XTestNCA, YTest, n_dimsNCA);
disp("Accuracy of KNN model (NCA with "+string(n_dimsNCA)+" features): " ...
    + string(accuracyknn));
disp("Precesion = "  + string(precisionknn));
disp("recall = " + string(recallknn));
disp("F-measure = " + string(F_measureknn));

% Evaluating with Decision tree
[accuracytree,precisiontree,recalltree,F_mesauretree] = treecperf(XTrainNCA,...
    YTrain, XTestNCA, YTest, n_dimsNCA);
disp("Accuracy of decision tree model (NCA with "+string(n_dimsNCA)+ ... 
 " features): " + string(accuracytree));
disp("Precesion = "  + string(precisiontree));
disp("recall = " + string(recalltree));
disp("F-measure = " + string(F_measuretree));

% Evaluating using Naive Bayes
[accuracynb,precisionnb,recallnb,F_measurenb] = nbcperf(XTrainNCA,...
    YTrain, XTestNCA, YTest, n_dimsNCA);
disp("Accuracy of Naive bayes model (NCA with "+string(n_dimsNCA)+...
" features): " + string(accuracynb));
disp("Precesion = "  + string(precisionnb));
disp("recall = " + string(recallnb));
disp("F-measure = " + string(F_measurenb));

%% Predicting the fault codes for actual test data
TestData = TestData(:,idx(1:n_dimsMRMR));
Model = fitctree(XTrainMRMR(:, 1:n_dimsMRMR), YTrain);
predictions = predict(Model,TestData);
FaultCode = predictions;
% converting fault codes into corresponding fault classes
for j = 1:1:20
    
    switch predictions(j,1)
        case 0
            predictions(j,1:3)=[0 0 0];
        case 1
            predictions(j,1:3)=[1 0 0];
        case 2
            predictions(j,1:3)=[0 1 0];
        case 3
            predictions(j,1:3)=[1 1 0];
        case 4
            predictions(j,1:3)=[0 0 1];
        case 5
            predictions(j,1:3)=[1 0 1];
        case 6
            predictions(j,1:3)=[0 1 1];
        case 7
            predictions(j,1:3)=[1 1 1];
    end 
end

% Converting results array into tabular form
N = (1:1:20)';
final_predictions = array2table([N,predictions,FaultCode],...
    'VariableNames',{'SNo','SensorDrift','ShaftWear','ToothFault','Fault code'});
disp("The final predictions are: ");
disp(final_predictions);


