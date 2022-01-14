# Fault-Classifier
Supervised machine learning model with 95% accuracy for detecting and classifying defects in an automobile transmission system. 

Applied feature selection (MRMR, NCA) and identified 5 key features out of 18 and used dimensionality reduction techniques (PCA, LDA) to optimize data. Trained KNN, Bayesian, Decision Tree classifiers, generated confusion charts and compared their performance metrics to select the model best suited for the task.

After the data is imported into the MATLAB workspace, it is split into training data and testing data. Following this, three classification models KNN, Naïve Bayes and Decision trees were trained, and the performance of the models was evaluated by calculating the accuracy. Next the training data is optimized through feature selection/dimensionality reduction. MRMR and NCA algorithms were used for feature selection to identify the features having the highest significance and PCA algorithm was used for dimensionality reduction. The above three classification models were again trained and tested using the optimized data. The accuracies and other performance metrics of the models are calculated. Finally, confusion charts of the models were developed and compared to select the best machine learning model for the task. 

