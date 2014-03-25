%%SSRF with DA from Leistner, Saffari et al. 2009 paper
%Authors: Luca Lonini and Sohrab Saeb
%Use modified Matlab TreeBagger class (X,Y properties set to public)

clear all
close all
tic;
%init
%an example dataset
load fisheriris
X = meas;   %predictors
Y = species; %labels
classes = unique(Y); %the classes
classcodes = (1:length(classes))';

%parameters
ntrees = 50;     %forest size
T0 = 1;          %initial temperature
alpha = .5;     %coeff to control the weight of the unlabeled part in the loss function
epochs = 20;     %epochs for unlabeled training
tau = epochs/2;  %cooling fcn time constant

%Randomly selects a portion of labeled data. Rest is Unlabeled
frac = 0.99;
labeled = zeros(size(X,1),1);
labeled(randperm(size(X,1),round(size(X,1)*frac))) = 1;
labeled = logical(labeled);

Xl = X(labeled,:);   Yl = Y(labeled);          %labeled data
Xu = X(~labeled,:);  Yu = cell(size(Xu,1),1);   %unlabeled data
Yu_forest = cell(size(ntrees*Xu,1),1);          %unlabeled data for entire forest


%%Train RF on labeled data
forest = TreeBagger(ntrees,Xl,Yl,'OOBPred','on');
oobind = forest.OOBIndices;
% OOBE = oobError(forest);  %out-of-bag error of the forest
%compute out-of-bag error for each tree
for t = 1:ntrees
    [Yp,Pl] = predict(forest,Xl(logical(oobind(:,t)),:),'trees', t);   %Pl is the prob of each tree
    acc = mean(strcmp(Yp,Yl(logical(oobind(:,t)))));
    GE(t) = 1-acc;  %generalization error for 1 tree
    
end
GEforest(1) = mean(GE);

% plot(GEforest)
% return;


%DA optimization
for m = 1:epochs
    T = T0*exp(-(m-1)/tau);     %reduce temp value
    
    [Yu_p,Pu] = predict(forest,Xu);    %compute prob Pu_i of each class (i) for the unlabeled data (output prob of forest)
    [r0,c0] = find(Pu == 0);      %find zero P values (to prevent log(0)=-inf)
    Pu(r0,c0) = eps;
    
    lg = log2(Pu).*Pu;  %margin max loss fcn (Entropy)
    
    %Compute Optimal Distribution over predicted labels
    Pu_opt = exp(-(alpha*lg + T)/T);
    Z = sum(Pu_opt,2); Z = repmat(Z,[1 3]);
    Pu_opt = Pu_opt./Z; %normalized probabilities
    
    
    %draw random label from Pu_opt distribution for each unlabeled data
    %point and each tree
    for p = 1:length(Yu)
        Yu_code = randsample(classcodes,ntrees,true,Pu_opt(p,:));  %predicted label for point p
        Yu_forest(p:length(Yu):length(Yu)*(ntrees-1)+p,1) = classes(Yu_code);
    end
    
     
    Xl_forest = repmat(Xl,[ntrees,1]); Xu_forest = repmat(Xu,[ntrees 1]);
    Yl_forest = repmat(Yl,[ntrees 1]); 
    X_forest = [Xl_forest;Xu_forest]; Y_forest = [Yl_forest;Yu_forest];
    
    %train forest on labeled and unlabeled data
    %Each tree is grown 
    forest = TreeBagger(ntrees,X_forest,Y_forest,'OOBPred','On','FBoot',1/ntrees);
    oobind = forest.OOBIndices;
    
    %find indices of out-of-bag labeled data
    OOBM = ones(length(Yl),ntrees);
    
    %produce oobindex for labeled data 
    for i = 1:ntrees
        OOBM_tmp = oobind((i-1)*length(Yl)+1:i*length(Yl),:);
        OOBM = OOBM.*OOBM_tmp;
    end
    
    %compute out-of-bag error for each tree
    for t = 1:ntrees
        [Yp,Pl] = predict(forest,Xl(logical(OOBM(:,t)),:),'trees', t);   %Pl is the prob of each tree
        acc = mean(strcmp(Yp,Yl(logical(OOBM(:,t)))));
        GE(t) = 1-acc;  %generalization error for 1 tree
%         if isnan(GE(t))
%             error('Nan found')
%         end
    end
    
    GEforest(m+1) = mean(GE);
    
        
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
plot(GEforest)


toc;