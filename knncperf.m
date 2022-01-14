function [accuracyknn,precisionknn,recallknn,F_measureknn] = knncperf(XTrain, YTrain, XTest, YTest, n_dims)
    
    %Fit model by assuming k=sqrt(n_samples)
    cknn = fitcknn(XTrain(:, 1:n_dims), YTrain, 'NumNeighbors', 37);
    
    %Make predictions
    predictionsknn = predict(cknn, XTest(:, 1:n_dims));
    
     % Evaluating model 
    TP = sum((string(predictionsknn) == string(YTest)) & YTest~=0);
    TN = sum((string(predictionsknn) == string(YTest)) & YTest==0);
    FP = sum((predictionsknn~=0) & (YTest==0));
    FN = sum((predictionsknn==0) & (YTest~=0));
    
    accuracyknn = sum(string(predictionsknn) == string(YTest))/length(YTest);
    precisionknn = TP/(TP+FP);
    recallknn = TP/(TP+FN);
    F_measureknn = (2*TP)/((2*TP) + FP + FN);
    
    %% Create confusion chart
    figure();
    cmknn = confusionchart(YTest, predictionsknn,...
        'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
    cmknn.Title = 'Knn confusion chart' ;
end