%%SSRF with DA from Leistner, Saffari et al. 2009 paper
%Authors: Luca Lonini and Sohrab Saeb
%Use modified Matlab TreeBagger class (X,Y properties set to public)

clear all;
close all
tic;
%init
%an example dataset
load g50c.mat;
% X = meas;   %predictors
% Y = label2code(species); %labels
Y = y/2+1.5;

n_class = length(unique(Y)); %the classes

%parameters
ntrees = 100;     %forest size
T0 = 5;          %initial temperature
alpha = 1;     %coeff to control the weight of the unlabeled part in the loss function
epochs = 200;     %epochs for unlabeled training
tau = 50;  %cooling fcn time constant
% frac = 0.99;

%Randomly selects a portion of labeled data. Rest is Unlabeled
% labeled = zeros(size(X,1),1);
% labeled(randperm(size(X,1),round(size(X,1)*frac))) = 1;
% labeled = logical(labeled);

% Xl = X(labeled,:);   Yl = Y(labeled);          %labeled data
% Xu = X(~labeled,:);  Yu = zeros(size(Xu,1),1);   %unlabeled data
Xl = X(idxLabs(1,:),:);    Yl = Y(idxLabs(1,:));
Xu = X(idxUnls(1,:),:);    Yu = Y(idxUnls(1,:));
Yu_forest = zeros(ntrees*size(Xu,1),1);          %unlabeled data for entire forest

%%Train RF on labeled data
forest = TreeBagger(ntrees,Xl,Yl,'OOBPred','on');
oobind = forest.OOBIndices;
%compute out-of-bag error for each tree
GE = zeros(ntrees,1);
Pl_forest = zeros(length(Yl),n_class);

% Computing the forest accuracy on unlabeled data
[Yfu,Pu_forest] = predict(forest,Xu);
Yfu = str2num(cell2mat(Yfu));
acc(1) = sum(Yu==Yfu)/length(Yu)

% return;

%DA optimization
lgs=[];
for m = 1:epochs
    T = T0*exp(-(m-1)/tau);     %reduce temp value
    Tvals(m) = T;               %save T values
    
    [Yu_p,Pu] = predict(forest,Xu);    %compute prob Pu_i of each class (i) for the unlabeled data (output prob of forest)
%     Yu_p = str2num(cell2mat(Yu_p));
%     [r0,c0] = find(Pu == 0);      %find zero P values (to prevent log(0)=-inf)
%     Pu(r0,c0) = eps;
%     Pu = Pu - .5;
    lg(:,1) = exp(-2*(Pu(:,1)-Pu(:,2)));  %margin max loss fcn (Entropy)
    lg(:,2) = exp(-2*(Pu(:,2)-Pu(:,1)));
    lgs = [lgs; mean(mean(lg))];
    
%     return;
    
    %Compute Optimal Distribution over predicted labels
    Pu_opt = exp(-(alpha*lg+T)/T);
    Z = sum(Pu_opt,2); Z = repmat(Z,[1 n_class]);
    Pu_opt = Pu_opt./Z; %normalized probabilities
    
    
    %draw random label from Pu_opt distribution for each unlabeled data
    %point and each tree
    for p = 1:length(Yu)
        Yu_temp = randsample(1:n_class,ntrees,true,Pu_opt(p,:));  %predicted label for point p
        Yu_forest(p:length(Yu):length(Yu)*(ntrees-1)+p,1) = Yu_temp;
    end
    
     
    Xl_forest = repmat(Xl,[ntrees,1]); Xu_forest = repmat(Xu,[ntrees 1]);
    Yl_forest = repmat(Yl,[ntrees 1]); 
    X_forest = [Xl_forest;Xu_forest]; Y_forest = [Yl_forest;Yu_forest];
    
    %train forest on labeled and unlabeled data
    %Each tree is grown 
    forest = TreeBagger(ntrees,X_forest,Y_forest,'OOBPred','On','FBoot',1/ntrees);
%     oob_tmp = oobError(forest);    
%     oob_total(m) = oob_tmp(end);    
    oobind = forest.OOBIndices;
    
    %find indices of out-of-bag labeled data
    OOBM = ones(length(Yl),ntrees);
    
    %produce oobindex for labeled data 
    for i = 1:ntrees
        OOBM_tmp = oobind((i-1)*length(Yl)+1:i*length(Yl),:);
        OOBM = OOBM.*OOBM_tmp;
    end
    
    %compute out-of-bag error for each tree
%     GE = [];
%     for t = 1:ntrees
%         if sum(OOBM(:,t))==0,
%             continue;
%         end
%         [Yp,Pl] = predict(forest,Xl(logical(OOBM(:,t)),:),'trees', t);   %Pl is the prob of each tree
%         Yp = str2num(cell2mat(Yp));
%         acc = mean((Yp-Yl(logical(OOBM(:,t))))==0);
%         GE = [GE, 1-acc];  %generalization error for 1 tree
%     end
  
    [Yfu,Pu_forest] = predict(forest,Xu);
    Yfu = str2num(cell2mat(Yfu));
    acc(m+1) = sum(Yu==Yfu)/length(Yu)

%     GEforest(m+1) = mean(GE);
    
        
    %compute predictions (scores or prob) for each out-of-bag datapoint
%     oobe = zeros(length(Yl),1);
%     for p = 1:length(Yl)
%         trees = find(OOBM(p,:) == 1);   %trees for which point p is out-of-bag
%         [Yp,Pl] = predict(forest,Xl(p,:),'trees', trees);   %Pl is the prob of each tree 
%         ind=find(ismember(classes,Yl(end)));                %the true class code
%         Ptrue = zeros(1,length(classes)); Ptrue(ind) = 1;   %the true prob vector for the class
%         oobe(p) = norm(Pl-Ptrue);
%     end
%     
%     oobetot(m) = mean(oobe);
end



figure
% plot(GEforest)
subplot(211), plot(acc);
subplot(212), plot(Tvals)

% figure(2)
% plot(lgs);

% figure(3);
% plot(oob_total);

toc;