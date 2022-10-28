clear variables; close all; clc;

%% From exercise1

%Remove undesirable warnings because of the comas
warning('off');

%Load the raw dataset in a table.
table = readtable('diabetes_dataset.csv');

%Enable the warnings again
warning('on');

%% Creation of the variables for a classification problem (1.5.1)

%Create the matrix X that contains the diverse attributes. We don't take
%diabetes outcome because we'll try to predict that.
% X_c = table2array(table(:,1:end));
X_c = table2array(table);

% Extract attribute names 
attributeNames_c = table.Properties.VariableNames(1:end-1)';

%% Visualize outliers

labels = {'N° of pregnancies','Plasma glucose concentration (mg/dl)','Diastolic blood pressure (mm Hg)',...
    'Skin Thickness (mm)','2 hours serum Insulin (µU/mL)','Body Mass Index (kg/m²)','Diabetes pedigree function (no unit)',...
    'Age (years)'};

f = figure();
f.WindowState = 'maximized';
for i = 1:8
    subplot(2,4,i)
    boxplot(X_c(:,i))
    ylabel(labels(i),'FontSize',16)
    xlabel('All observations','FontSize',16)
end
    
%% Then we remove the outliers as discussed

%First remove irrelevant 0 for glucose (2), blood pressure (3), skin
%thickness (4), insulin (5) and BMI (6). These  are corrupted values.

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




% These steps are not needed because we only have 2 classes (diabetic or
% not) which are not text as in exercise 1.
classLabels = X_c(:,9); 
classNames = {'non-diabetic', 'diabetic'};
% [~,y] = ismember(classLabels, classNames);

%Now we can remove the last column
X_c = X_c(:,1:end-1);

%Instead, we directly compute y as the last column:
y_c = table2array(table(:,9));
y_c = y_c(corrupted_lines);
y_c = y_c(outliers);

%Compute the size of X. M is the number of observations. N the number of
%attributes
[N_c, M_c] = size(X_c);

%and the number of class
C = length(unique(y_c));



%% Plotting scatter plots 


f = figure();
% Use a specific color for each class
colors = get(gca, 'colororder'); 

%Multiple plots
%Choose the attributes to be plotted
% A = 8;
% for i = 1:8
%     if i ~= A
%         subplot(2,4,i)
%         gscatter(X_c(:,A), X_c(:,i), classLabels)
%         axis tight
%         legend(classNames);
%         xlabel(attributeNames_c{A});
%         ylabel(attributeNames_c{i});
%     end
% end

%Single plot
%Choose the attributes to be plotted
A=1;
B=2;
hold all
gscatter(X_c(:,A), X_c(:,B), classLabels)
axis tight
legend(classNames);
xlabel(attributeNames_c{A});
ylabel(attributeNames_c{B});
title('Diabetes data');

%% Data visualization

f= figure();
f.WindowState = 'maximized';
for i = 1:8
    subplot(2,4,i)
    histfit(X_c(:,i),17,'normal')
%     hold on
    xlabel(attributeNames_c(i),'FontSize',16)
%     mean_i = mean(X_c(:,i));
%     std_i = std(X_c(:,i));
    
end

%% Correlation map
f = figure();
corrplot(X_c,'varNames',attributeNames_c)

%% Covariance matrix
cov_mat = cov(X_c);

%% Principal component analysis

% Subtract the mean from the data
Y = bsxfun(@minus, X_c, mean(X_c));
Y = bsxfun(@times, Y, 1./std(X_c));

% Obtain the PCA solution by calculate the SVD of Y
[U, S, V] = svd(Y);

% Compute variance explained
rho = diag(S).^2./sum(diag(S).^2);
threshold = 0.90;

% Plot variance explained
f = figure();
hold on
plot(rho, 'x-');
plot(cumsum(rho), 'o-');
plot([0,length(rho)], [threshold, threshold], 'k--');
legend({'Individual','Cumulative','Threshold'}, ...
        'Location','best');
ylim([0, 1]);
xlim([1, length(rho)]);
grid minor
xlabel('Principal component');
ylabel('Variance explained value');
title('Variance explained by principal components');


% Index of the principal components
PC_1 = 1;
PC_2 = 2;
PC_3 = 3;

% Compute the projection onto the principal components
Z = U*S;

% Plot PCA of data
f = figure();
hold all
colors = get(gca,'colororder');
for c = 0:C-1
    scatter(Z(y_c==c,PC_1), Z(y_c==c,PC_2), 50,  ...
                'MarkerFaceColor', colors(c+1,:), ...
                'MarkerEdgeAlpha', 0, ...
                'MarkerFaceAlpha', .5);
end
legend(classNames);
axis tight
xlabel(sprintf('PC %d', PC_1));
ylabel(sprintf('PC %d', PC_2));
title('PCA Projection of Diabetes data');

%% test 3d

f = figure();
colors = get(gca,'colororder');
scatter3(Z(y_c==0,PC_1), Z(y_c==0,PC_2),Z(y_c==0,PC_3),50,'MarkerEdgeColor','b') 
hold on
scatter3(Z(y_c==1,PC_1), Z(y_c==1,PC_2),Z(y_c==1,PC_3),50,'MarkerEdgeColor','r') 
hold off
legend(classNames);
axis tight
xlabel(sprintf('PC %d', PC_1));
ylabel(sprintf('PC %d', PC_2));
zlabel(sprintf('PC %d', PC_3));
title('PCA Projection of Diabetes data');

% test

% f= figure();
% colors = get(gca,'colororder');
% data1 = Z(y_c==0,PC_1);      
% data2 = Z(y_c==1,PC_1);  
% hAxes = axes('NextPlot','add',...            
%              'DataAspectRatio',[1 1 1],...  
%              'XLim',[-4.5 4.5],...               
%              'YLim',[0 eps],...               
%              'Color','none');               
% plot(data1,0,'b.','MarkerSize',12);  
% % legend('non-diabetic')
% hold on
% plot(data2,0,'r.','MarkerSize',12);  
% legend('non-diabetic','diabetic')

%% Direction

z = zeros(1,size(V,2))';
f = figure();
quiver(z,z,V(:,PC_1), V(:,PC_2), 1, ...
               'Color', 'k', ...
                'AutoScale','off', ...
               'LineWidth', .1)
hold on
for pc=1:length(attributeNames_c)
      text(V(pc,PC_1), V(pc,PC_2),attributeNames_c{pc}, ...
                 'FontSize', 10)
 end
        xlabel('PC1')
        ylabel('PC2')
        grid; box off; axis equal;
        % Add a unit circle
        plot(cos(0:0.01:2*pi),sin(0:0.01:2*pi));
        axis tight

%% Creation of the variables for a regression problem (1.5.4)

% %Store the data
% data = [X_c,y_c];
% 
% %We wish to predict the BMI which is in the 6th column
% y_r = data(:,6);
% 
% %Then we remove it from the initial matrix
% X_r = [data(:,1:5),data(:,7:end)];
% 
% %Compute the size
% [N_c,M_c] = size(X_r);


%% 