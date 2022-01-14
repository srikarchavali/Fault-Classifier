function [accuracytree,precisiontree,recalltree,F_measuretree] = treecperf(XTrain, YTrain, XTest, YTest, n_dims)
   
    % Using decision tree to fit model
    ctree = fitctree(XTrain(:, 1:n_dims), YTrain);
    
    %Make predictions
    predictionstree = predict(ctree, XTest(:, 1:n_dims));
    
    % Evaluating model 
    TP = sum((string(predictionstree) == string(YTest)) & YTest~=0);
    TN = sum((string(predictionstree) == string(YTest)) & YTest==0);
    FP = sum((predictionstree~=0) & (YTest==0));
    FN = sum((predictionstree==0) & (YTest~=0));
    
    %Evaluate results
    accuracytree = sum(string(predictionstree) == string(YTest))/length(YTest);
    precisiontree = TP/(TP+FP);
    recalltree = TP/(TP+FN);
    F_measuretree = (2*TP)/((2*TP) + FP + FN);
   
    % Create confusion chart
    figure();
    cmtree = confusionchart(YTest, predictionstree,...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
    cmtree.Title = 'Decision Tree confusion chart';
   
    
end