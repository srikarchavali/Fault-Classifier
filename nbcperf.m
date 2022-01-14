function [accuracynb,precisionnb,recallnb,F_measurenb] = nbcperf(XTrain, YTrain, XTest, YTest, n_dims)
   
    % Using Naive bayes to fit model
    cnb = fitcnb(XTrain(:, 1:n_dims), YTrain);
    
    %Make predictions
    predictionsnb = predict(cnb, XTest(:, 1:n_dims));
    
    % Evaluating model 
    TP = sum((string(predictionsnb) == string(YTest)) & YTest~=0);
    TN = sum((string(predictionsnb) == string(YTest)) & YTest==0);
    FP = sum((predictionsnb~=0) & (YTest==0));
    FN = sum((predictionsnb==0) & (YTest~=0));
    
    accuracynb = sum(string(predictionsnb) == string(YTest))/length(YTest);
    precisionnb = TP/(TP+FP);
    recallnb = TP/(TP+FN);
    F_measurenb = (2*TP)/((2*TP) + FP + FN);
    
    %% Create confusion chart
    figure();
     cmnb = confusionchart(YTest, predictionsnb,...
         'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
     cmnb.Title = 'Naive Bayes confusion chart' ;
end