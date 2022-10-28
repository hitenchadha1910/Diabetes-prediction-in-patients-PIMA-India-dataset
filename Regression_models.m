% clear variables; close all; clc;

%% Report 2: Classification and regression on a Diabetes data set.

%Remove undesirable warnings because of the comas
warning('off');

%Load the raw dataset in a table.
table = readtable('diabetes_dataset.csv');

%Enable the warnings again
warning('on');

%% Regression - part a 

%Create the matrix
X_c = table2array(table);

% Extract attribute names 
% attributeNames_c = table.Properties.VariableNames(1:end-1)';
attributeNames_c = table.Properties.VariableNames(1:end)';

%Compute the columns that contain corrupted values
corrupted = X_c(:,2:6);

%Compute a vector that contains a 0 if the line contains at least one 0
corrupted_lines = all( corrupted,2 );

%Keep only lines with no corrupted values
X_c = X_c( corrupted_lines , : );

%Then we remove outliers for blood pressure. There does not seem to be
%blood pressure that appear to high. We'll remove blood pressure below 50
%mm/Hg that are impossible for alive humans. Blood pressure is index 3
BP = X_c(:,3);
outliers = all( BP > 50 , 2);
X_c = X_c(outliers,:);

classLabels = X_c(:,9); 
classNames = {'non-diabetic', 'diabetic'};

%Now we can remove the last column
% X_c = X_c(:,1:end-1);

%Instead, we directly compute y as the last column:
% y_c = table2array(table(:,9));
% y_c = y_c(corrupted_lines);
% y_c = y_c(outliers);
% 
% %% Regression part a
% 
% % The first thing we do is store all the information we have in the
% % other format in one data matrix:
% data = [X_c,y_c];


% We know that the BMI corresponds to the sixth column in the data
% matrix (see attributeNames), and therefore our new y variable is:
y_r = X_c(:, 6);
%y_r = X_c(:, 4);


%Then we remove it from the initial matrix
X_r = [X_c(:,1:5),X_c(:,7:end)];

%Compute the size
[N,M] = size(X_r);

%The following is modified from script 8_1_1 mainly
%Also used Algorithm 5 in the book p173 in chapter 10

%% Regression part a and b 
% Create crossvalidation partition for evaluation
K = 5;

%statistics
% rng(1);
m = 2; %number of repeats
J = K*m; %Total number of splits
r_LinR_baseline = nan(J,1);
r_LinR_ANN = nan(J,1);
r_ANN_baseline = nan(J,1);

for dm=1:m
% Initialize variables
lambda =10.^(-6:3);

% N_units = (1:10);

% Initialize variables
T=length(lambda);
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_rlr = nan(K,1);
Error_test_rlr = nan(K,1);
w = nan(M,T,K);
lambda_opt = nan(K,1);
w_rlr = nan(M,K);
mu = nan(K, 1);
sigma = nan(K, 1);

% Variables for regression error in ANN
Error_train_ann = nan(K,1);
Error_test_ann = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);



CV = cvpartition(N, 'Kfold', K);

for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    % Extract the training and test set
    X_train = X_r(CV.training(k), :);
    y_train = y_r(CV.training(k));
    X_test = X_r(CV.test(k), :);
    y_test = y_r(CV.test(k));
    
    
    KK = 5;
    CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
      
    for kk=1:KK
        % Extract the training and test set for regression
        X_train2 = X_train(CV2.training(kk), :);
        y_train2 = y_train(CV2.training(kk));
        X_test2 = X_train(CV2.test(kk), :);
        y_test2 = y_train(CV2.test(kk));
    
        %standardisation
        mu = mean(X_train2,1);
        sigma = std(X_train2,1);
        X_train2 = bsxfun(@times, X_train2 - mu, 1./ sigma);
        X_test2 = bsxfun(@times, X_test2 - mu, 1./ sigma);

%         Xty = X_train2' * y_train2;
%         XtX = X_train2' * X_train2;

        % Parameters for neural network classifier
        NHiddenUnits = (5:15);  % Number of hidden units
        NTrain = 2; % Number of re-trains of neural network
    
        for t=1:length(lambda)
            mdl = fitrlinear(X_train2, y_train2, ...
                     'Lambda', lambda(t), ...
                     'Learner', 'leastsquares', ...
                     'Regularization', 'ridge');
                y_train_est = predict(mdl, X_train2);
    %             train_error_rate(t,kk) = sum( y_train ~= y_train_est ) / length(y_train);
                Error_train(t,kk) = sum((y_train2-y_train_est).^2)/sum(CV2.TrainSize);

                y_test_est = predict(mdl, X_test2);
    %             test_error_rate(t,kk) = sum( y_test ~= y_test_est ) / length(y_test);
                Error_test(t,kk) = sum((y_test2-y_test_est).^2)/sum(CV2.TestSize);
   
        end
    
        % Extract training and test set for ANN
        X_train_ann = X_train2;
        y_train_ann = y_train2;
        X_test_ann = X_test2;
        y_test_ann = y_test2;
        
        
        % Fit neural network to training set with nr_main function in tools
        MSEBest = inf;
        for N_hidden = 1:length(NHiddenUnits)            
            for t = 1:NTrain
                netwrk = nr_main(X_train_ann, y_train_ann, X_test_ann, y_test_ann, NHiddenUnits(N_hidden));
                if netwrk.mse_train(end)<MSEBest
                   bestnet{kk} = netwrk; 
                   MSEBest=netwrk.mse_train(end); 
                end
            end
            
            
            
        end
    
        % Predict model on test and training data    
        y_train_ann_est = bestnet{kk}.t_pred_train;    
        y_test_ann_est = bestnet{kk}.t_pred_test;        
    
        % Compute least squares error
        Error_train_ann(kk) = sum((y_train_ann-y_train_ann_est).^2)/sum(CV2.TrainSize);
        Error_test_ann(kk) = sum((y_test_ann-y_test_ann_est).^2)/sum(CV2.TestSize); 



        %Display best Neural Network
%         displayNetworkRegression(bestnet{kk});
    end
    
    %Optimal Linear Model

    [min_val, lambda_idx] = min(sum(Error_test,2));
    lambda_opt(k) = lambda(lambda_idx);

    X_opt = X_train;
    y_opt = y_train;
    mu_opt = mean(X_opt,1);
    std_opt = std(X_opt,1);
    X_opt = bsxfun(@times, X_opt - mu_opt, 1./ std_opt);
    X_test = bsxfun(@times, X_test - mu_opt, 1./ std_opt);

%     Xty = X_opt' * y_train;
%     XtX = X_opt' * X_opt;

    mdl_opt = fitrlinear(X_opt, y_opt, ...
                'Lambda', lambda_opt(k), ...
                 'Learner', 'leastsquares', ...
                 'Regularization', 'ridge');
    y_train_est = predict(mdl_opt, X_opt);
    Error_train_rlr(k) = sum((y_train-y_train_est).^2)/sum(CV.TrainSize);
    %Test error
    y_test_est = predict(mdl_opt, X_test);
    Error_test_rlr(k) = sum((y_test-y_test_est).^2)/sum(CV.TestSize);
    
    % Compute least squares error when predicting output to be mean of
    % training data- Baseline model
%     Error_train_nofeatures(k) = sum((y_train_ann-mean(y_train_ann)).^2)/length(y_train_ann);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2)/sum(CV.TestSize);
    
    
    % calculate average difference in loss for this fold. 
    r_LinR_baseline((dm-1)*K+k) = (Error_test_rlr(k)-Error_test_nofeatures(k));
    r_LinR_ANN((dm-1)*K+k) = (Error_test_rlr(k)-Error_test_ann(k));
    r_ANN_baseline((dm-1)*K+k) = (Error_test_ann(k)- Error_test_nofeatures(k));
    
end
end

% lambda_opt = 100;
% regularization_opt = lambda_opt * eye(M);
% regularization_opt(1,1) = 0; % Remove regularization of bias-term
% 
% % w_opt = nan(N,1);
% w_opt=(XtX+regularization_opt)\ Xty; %EQUATION FROM THE EXERCISE INSTRUCTIONS BELOW
% 
% % evaluate training and test error performance for optimal selected value of
% % lambda
% Error_train_rlr = sum((y_train-X_opt*w_opt).^2);
% Error_test_rlr = sum((y_test-X_test*w_opt).^2);

%test = predict(mdl_opt,X_test);
%diff = y_test-test;

%% Statistics
alpha = 0.05;
rho = 1/K;
[p_LinR_baseline, CI_LinR_baseline] = correlatedttest(r_LinR_baseline, rho, alpha);
[p_LinR_ANN, CI_LinR_ANN] = correlatedttest(r_LinR_ANN, rho, alpha);
[p_ANN_baseline, CI_ANN_baseline] = correlatedttest(r_ANN_baseline, rho, alpha);

%% Plotting for linear regression

f = figure();
f.WindowState = 'maximized';
x_LR= lambda;
semilogx(x_LR,mean(Error_train,2),'LineWidth',3)
hold on
semilogx(x_LR,mean(Error_test,2),'LineWidth',3)
yline(mean(Error_test_nofeatures),'--r','LineWidth',4);
title(['Optimal value of lambda: 1e' num2str(log10(lambda_opt(end)))],'FontSize',14);
legend({'Training error','Test error','Baseline'},'FontSize',14)
xlabel('Regularization strength lambda','FontSize',14)
ylabel('Error rate','FontSize',14)
% ylim([-1 1])
% xlim([10e-4 10e2])

%%
ANN_numbers = [bestnet{1}(1).Nh(1), bestnet{2}(1).Nh(1), bestnet{3}(1).Nh(1), bestnet{4}(1).Nh(1), bestnet{5}(1).Nh(1)];
table = [ANN_numbers' Error_test_ann lambda_opt Error_test_rlr Error_test_nofeatures];
table_avg = mean(table,1);
    
    