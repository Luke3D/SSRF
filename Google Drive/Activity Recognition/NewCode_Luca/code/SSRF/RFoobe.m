load fisheriris
X = meas;   %predictors
Y = species; %labels
classes = unique(Y); %the classes
classcodes = (1:length(classes))';
ntrees = 50;

Xl = X; Yl = Y;

forest = TreeBagger(ntrees,Xl,Yl,'OOBPred','On','FBoot',1);
oobind = forest.OOBIndices;

% [Yp,Pl] = predict(forest,Xl);
plot(oobError(forest))

OOBM = oobind;

%compute predictions (scores or prob) for each out-of-bag datapoint
oobe = zeros(length(Yl),1);
for t = 1:ntrees
    [Yp,Pl] = predict(forest,Xl(logical(oobind(:,t)),:),'trees', t);   %Pl is the prob of each tree
    acc = mean(strcmp(Yp,Yl(logical(oobind(:,t)))));    
    GE(t) = 1-acc;  %generalization error for 1 tree
    
    
%     ind=find(ismember(classes,Yl));                %the true class code
%     Plvec = Pl==max(Pl);
%     Ptrue = zeros(1,length(classes)); Ptrue(ind) = 1;   %the true prob vector for the class
%     oobe (p) = max(Plvec - Ptrue);
    %oobe(p) = norm(Pl-Ptrue);
end

GEforest = mean(GE)

% for p = 1:length(Yl)
%     trees = find(OOBM(p,:) == 1);   %trees for which point p is out-of-bag
%     [Yp,Pl] = predict(forest,Xl(p,:),'trees', trees);   %Pl is the prob of each tree
%     ind=find(ismember(classes,Yl(end)));                %the true class code
%     Plvec = Pl==max(Pl);
%     Ptrue = zeros(1,length(classes)); Ptrue(ind) = 1;   %the true prob vector for the class
%     oobe (p) = max(Plvec - Ptrue);
%     %oobe(p) = norm(Pl-Ptrue);
% end
