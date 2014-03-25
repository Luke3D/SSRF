%% class for SSRF

classdef ssforest < handle
    properties
        
        ntrees;
        T0;          %initial temperature
        alpha;     %coeff to control the weight of the unlabeled part in the loss function
        tau;       %cooling fcn time constant
        Xl;
        Yl;
        Xu;
        Yu;
        n_class;
        acc;
        Tvals;
    end
    
    methods
        function obj = ssforest(PARAM)
            
            obj.ntrees = PARAM{1};
            obj.T0 = PARAM{2};
            obj.alpha = PARAM{3};
            obj.tau = PARAM{4};
            obj.acc = [];
            obj.Tvals = []; 
            
            %dataset
            obj.Xl = PARAM{5}; obj.Yl = PARAM{6};
            obj.Xu = PARAM{7}; obj.Yu = PARAM{8};
            obj.n_class = PARAM{9};
            
        end
        
        function [acc, Tvals] = trainforest(this,epochs)
            
            
            %labeled and unlabeled data
            
            Xl = this.Xl; Yl = this.Yl;
            Xu = this.Xu; Yu = this.Yu;
            
            %%Train RF on labeled data
            forest = TreeBagger(this.ntrees,Xl,Yl,'OOBPred','on');
            oobind = forest.OOBIndices;
            %compute out-of-bag error for each tree
            GE = zeros(this.ntrees,1);
            Pl_forest = zeros(length(Yl),this.n_class);
            
            % Computing the forest accuracy on unlabeled data
            [Yfu,Pu_forest] = predict(forest,Xu);
            Yfu = str2num(cell2mat(Yfu));
            acc(1) = sum(Yu==Yfu)/length(Yu)
            
            % return;
            
            %DA optimization
            lgs=[];
            for m = 1:epochs
                T = this.T0*exp(-(m-1)/this.tau);     %reduce temp value
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
                Pu_opt = exp(-(this.alpha*lg+T)/T);
                Z = sum(Pu_opt,2); Z = repmat(Z,[1 this.n_class]);
                Pu_opt = Pu_opt./Z; %normalized probabilities
                
                
                %draw random label from Pu_opt distribution for each unlabeled data
                %point and each tree
                for p = 1:length(Yu)
                    Yu_temp = randsample(1:this.n_class,this.ntrees,true,Pu_opt(p,:));  %predicted label for point p
                    Yu_forest(p:length(Yu):length(Yu)*(this.ntrees-1)+p,1) = Yu_temp;
                end
                
                
                Xl_forest = repmat(Xl,[this.ntrees,1]); Xu_forest = repmat(Xu,[this.ntrees 1]);
                Yl_forest = repmat(Yl,[this.ntrees 1]);
                X_forest = [Xl_forest;Xu_forest]; Y_forest = [Yl_forest;Yu_forest];
                
                %train forest on labeled and unlabeled data
                %Each tree is grown
                forest = TreeBagger(this.ntrees,X_forest,Y_forest,'OOBPred','On','FBoot',1/this.ntrees);
                %     oob_tmp = oobError(forest);
                %     oob_total(m) = oob_tmp(end);
                oobind = forest.OOBIndices;
                
                %find indices of out-of-bag labeled data
                OOBM = ones(length(Yl),this.ntrees);
                
                %produce oobindex for labeled data
                for i = 1:this.ntrees
                    OOBM_tmp = oobind((i-1)*length(Yl)+1:i*length(Yl),:);
                    OOBM = OOBM.*OOBM_tmp;
                end
                
                [Yfu,Pu_forest] = predict(forest,Xu);
                Yfu = str2num(cell2mat(Yfu));
                acc(m+1) = sum(Yu==Yfu)/length(Yu)
                
            end
            
            figure
            subplot(211), plot(acc);
            subplot(212), plot(Tvals)
            
            this.acc = acc;
            this.Tvals = Tvals;
        end
        
        
        
    end
    
end






