function forest = config()

%parameters
ntrees = 100;     %forest size
T0 = 5;          %initial temperature
alpha = 0.1;     %coeff to control the weight of the unlabeled part in the loss function
tau = 25;  %cooling fcn time constant


%dataset
load g50c.mat;
% X = meas;   %predictors
% Y = label2code(species); %labels
Y = y/2+1.5;

n_class = length(unique(Y)); %the classes
%Randomly selects a portion of labeled data. Rest is Unlabeled
% labeled = zeros(size(X,1),1);
% labeled(randperm(size(X,1),round(size(X,1)*frac))) = 1;
% labeled = logical(labeled);
% Xl = X(labeled,:);   Yl = Y(labeled);          %labeled data
% Xu = X(~labeled,:);  Yu = zeros(size(Xu,1),1);   %unlabeled data

Xl = X(idxLabs(1,:),:);    Yl = Y(idxLabs(1,:));
Xu = X(idxUnls(1,:),:);    Yu = Y(idxUnls(1,:));

PARAM = {ntrees,T0,alpha,tau,Xl,Yl,Xu,Yu,n_class};

forest = ssforest(PARAM);

end