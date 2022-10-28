clear variables; close all; clc;

%% From exercise1

%Remove undesirable warnings because of the comas
warning('off');

%Load the raw dataset in a table.
table = readtable('diabetes_dataset.csv');

%Enable the warnings again
warning('on');

%% Creation of the variables for a classification problem (1.5.1)

%Create the matrix X that contains the diverse attributes.
X = table2array(table);

% Extract attribute names 
attributeNames_c = table.Properties.VariableNames(1:end-1)';

%% Then we remove the outliers as discussed

%First remove irrelevant 0 for glucose (2), blood pressure (3), skin
%thickness (4), insulin (5) and BMI (6). These  are corrupted values.

%Compute the columns that contain corrupted values
corrupted = X(:,2:6);

%Compute a vector that contains a 0 if the line contains at least one 0
corrupted_lines = all( corrupted,2 );

%Keep only lines with no corrupted values
X = X( corrupted_lines , : );

%Then we remove outliers for blood pressure. There does not seem to be
%blood pressure that appear to high. We'll remove blood pressure below 50
%mm/Hg that are impossible for alive humans. Blood pressure is index 3
BP = X(:,3);
outliers = all( BP > 50 , 2);
X = X(outliers,:);

% These steps are not needed because we only have 2 classes (diabetic or
% not) which are not text as in exercise 1.
classLabels = X(:,9); 
classNames = {'non-diabetic', 'diabetic'};
% [~,y] = ismember(classLabels, classNames);

%Now we can remove the last column
X = X(:,1:end-1);

%Instead, we directly compute y as the last column:
y = table2array(table(:,9));
y = y(corrupted_lines);
y = y(outliers);

%Compute the size of X. M is the number of observations. N the number of
%attributes
[N, M] = size(X);

%and the number of class
C = length(unique(y));

%% Logistic regression: combined ex 8_1_1 and 8_1_2



% Crossvalidation outer layer
% Create crossvalidation partition for evaluation of performance of optimal
% model
K = 10;
%statistics
rng(1);
m = 3; %number of repeats
J = K*m; %Total number of splits
r_KNN_baseline = nan(J,1);
r_KNN_LR = nan(J,1);
r_LR_baseline = nan(J,1);

for dm=1:m
CV = cvpartition(N, 'Kfold', K);

% Values of lambda
% lambda = logspace(-8,0,50);
lambda = 10.^(-8:5);

% Initialize variables
T = length(lambda);
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train2 = nan(T,K);
Error_test2 = nan(T,K);
lambda_opt = nan(K,1);
% mu = nan(K, M-1);
% sigma = nan(K, M-1);
table_value = nan(K,2);
table_knn = nan(K,2);
error_baseline = nan(K,1);

KNNmax = 50;                    %Maximum number of neighbors
Distance = 'euclidean'; % Distance measure
% Distance = 'cityblock'; % Distance measure
% Distance = 'correlation'; % Distance measure
% Distance = 'cosine'; % Distance measure
E_train_KNN = nan(K,1);
E_test_KNN = nan(K,1);
E_train_KNN2 = nan(KNNmax,K);
E_test_KNN2 = nan(KNNmax,K);
N_Neighbor_opt = nan(K,1);

test_error_rate = nan(length(lambda),K);
train_error_rate = nan(length(lambda),K);
coefficient_norm = nan(length(lambda),K);

% For each crossvalidation fold (outer)
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    

    
    % Use 10-fold crossvalidation to estimate optimal value of lambda    
    KK = 10;
    CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
    
    %Find the optimal model
	for kk=1:KK
        X_train2 = X_train(CV2.training(kk), :);
        y_train2 = y_train(CV2.training(kk));
        X_test2 = X_train(CV2.test(kk), :);
        y_test2 = y_train(CV2.test(kk));
        
        %Save these variables for KNN classification since we won't
        %standardize here
%         X_train2_KNN = X_train2;
%         y_train2_KNN = y_train2;
%         X_test2_KNN = X_test2;
%         y_test2_KNN = y_test2;

        
        % Standardize the training and test set based on training set in
        % the inner fold
        mu2 = mean(X_train2,1);
        sigma2 = std(X_train2,1);
        X_train2 = bsxfun(@times, X_train2 - mu2, 1./ sigma2);
        X_test2 = bsxfun(@times, X_test2 - mu2, 1./ sigma2);
        
        for t = 1:length(lambda)
            mdl = fitclinear(X_train2, y_train2, ...
                 'Lambda', lambda(t), ...
                 'Learner', 'logistic', ...
                 'Regularization', 'ridge');
            [y_train_est, p] = predict(mdl, X_train2);
%             train_error_rate(t,kk) = sum( y_train ~= y_train_est ) / length(y_train);
            Error_train2(t,kk) = sum( y_train2 ~= y_train_est ) / length(y_train2);
    
            [y_test_est, p] = predict(mdl, X_test2);
%             test_error_rate(t,kk) = sum( y_test ~= y_test_est ) / length(y_test);
            Error_test2(t,kk) = sum( y_test2 ~= y_test_est ) / length(y_test2);    
        end
        
        for n_neighbor = 1:KNNmax
            knn=fitcknn(X_train2, y_train2, 'NumNeighbors', n_neighbor, 'Distance', Distance);
            y_test_est=predict(knn, X_test2);
            y_train_est = predict(knn, X_train2);
            % Compute number of classification errors
            E_test_KNN2(n_neighbor,kk) = sum(y_test2 ~=y_test_est) / length(y_test2); % Count the number of errors
            E_train_KNN2(n_neighbor,kk) = sum(y_train2 ~= y_train_est) / length(y_train2);
        end
        
    end
   [min_error,lambda_idx] = min(sum(Error_test2,2)/sum(CV2.TestSize));
   lambda_opt(k)=lambda(lambda_idx);  
   
   [min_knn,neighbor_idx] = min(sum(E_test_KNN2,2)/sum(CV2.TestSize));
   N_Neighbor_opt(k) = neighbor_idx;
   
    % Standardize datasets in outer fold, and save the mean and standard
    % deviations since they're part of the model (they would be needed for
    % making new predictions)
    mu(k,  :) = mean(X_train);
    sigma(k, :) = std(X_train);
    X_train_std = (X_train - mu(k , :)) ./ sigma(k, :);
    X_test_std = (X_test - mu(k, :)) ./ sigma(k, :);
        
    
    %Create model for optimal value of lambda
    mdl_opt = fitclinear(X_train_std, y_train, ...
                 'Lambda', lambda_opt(k), ...
                 'Learner', 'logistic', ...
                 'Regularization', 'ridge');
                         
    % evaluate training and test error performance for optimal selected value of
    % lambda 
    %Train error
    [y_train_est, p] = predict(mdl_opt, X_train_std);
    Error_train(k) = sum( y_train ~= y_train_est ) / length(y_train);
    %Test error
	[y_test_est, p] = predict(mdl_opt, X_test_std);
    Error_test(k) = sum( y_test ~= y_test_est ) / length(y_test);
    
    %Create model for optimal value of neighbors
    knn_opt=fitcknn(X_train_std, y_train, 'NumNeighbors', N_Neighbor_opt(k), 'Distance', Distance);
    
    %Evaluate training and test error for the optimal selected value.
    y_test_est=predict(knn_opt, X_test_std);
    y_train_est = predict(knn_opt, X_train_std);
	% Compute number of classification errors
	E_test_KNN(k) = sum(y_test ~=y_test_est) / length(y_test); % Count the number of errors
	E_train_KNN(k) = sum(y_train ~= y_train_est) / length(y_train);
    
    %baseline
    y_baseline = ones(size(y_test));
    error_baseline(k) = sum( y_baseline ~= y_test) / length(y_test);
    
    % calculate average difference in loss for this fold. 
    r_KNN_baseline((dm-1)*10+k) = (E_test_KNN(k)-error_baseline(k))/length(y_test);
    r_KNN_LR((dm-1)*10+k) = (E_test_KNN(k)-Error_test(k))/length(y_test);
    r_LR_baseline((dm-1)*10+k) = (Error_test(k)-error_baseline(k))/length(y_test);
    
    table_value(k,1) = lambda_opt(k);
    table_value(k,2) = Error_test(k);
    table_knn(k,1) = N_Neighbor_opt(k);
    table_knn(k,2) = E_test_KNN(k);

   
end
end


table = [table_knn table_value error_baseline];
rounded = round(table,2);

%% Statistics
alpha = 0.05; 
rho = 1/K; 
[p_KNN_baseline, CI_KNN_baseline] = correlatedttest(r_KNN_baseline, rho, alpha);
[p_KNN_LR, CI_KNN_LR] = correlatedttest(r_KNN_LR, rho, alpha);
[p_LR_baseline, CI_LR_baseline] = correlatedttest(r_LR_baseline, rho, alpha);

%% Plotting for KNN

f = figure();
f.WindowState = 'maximized';
x_KNN = (1:KNNmax);
plot(x_KNN,mean(E_train_KNN2,2)*100)
hold on
plot(x_KNN,mean(E_test_KNN2,2)*100)
yline(error_baseline(end)*100,'--r','LineWidth',5);
title('Training and test error in function of the number of neighbors in KNN classification (averaged over all 10 inner folds)');
legend({'Training error','Test error','Baseline'})
xlabel('Number of Neighbors')
ylabel('Error rate [%]')
ylim([10 70])


%% Plotting for logistic regression

f = figure();
f.WindowState = 'maximized';
x_LR= lambda;
semilogx(x_LR,mean(Error_train2,2)*100)
hold on
semilogx(x_LR,mean(Error_test2,2)*100)
yline(error_baseline(end)*100,'--r','LineWidth',5);
title('Training and test error in function of the regularization strength (averaged over all 10 inner folds)');
legend({'Training error','Test error','Baseline'})
xlabel('Lambda')
ylabel('Error rate [%]')
ylim([15 70])
xlim([10e-4 10e2])

    