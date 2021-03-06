 load 'RWD.MAT';
 
 %trainData = [class_var(1:400,:); class_var(501:900,:); class_var(1001:1400,:)];
 %expData = [class_var(401:500,:); class_var(901:1000,:); class_var(1401:1500,:)];

 %trainData = [ls_group3(1:400,:); ls_group3(501:900,:); ls_group3(1001:1400,:);ls_group3(1501:1900,:)];
 %expData = [ls_group3(401:500,:); ls_group3(901:1000,:); ls_group3(1401:1500,:);ls_group3(1901:2000,:)];

 
 trainData = [RWD(1:1000,:); RWD(2292:3291,:); RWD(4456:5455,:)];
 expData = [RWD(1001:2291,:); RWD(3292:4455,:); RWD(5455:6943,:)];

 clf;
 set(gcf,'Name','Decision Boundaries'); 
 for cases=1:5
     subplot(2,3,cases);
 model = BuildBaysianModel(trainData,0,cases); %change case here
 %expResults = BayesianClassify(model,expData(:,1:2));

 m = size(expData, 1); % number of examples
 n = size(expData, 2); % number of feature dimension
 k = length(unique(trainData(:, end)));

 DETLabels  = zeros(m,10);

 for l = 1:m
    p = zeros(k);
    
    for i=1:k
        p(i) = (1/((2*pi)^(-n/2) * det(cell2mat(model(i,2)))^0.5)) * exp(-0.5*(bsxfun(@minus,expData(l,1:2),cell2mat(model(i,1))))*inv(cell2mat(model(i,2)))*(bsxfun(@minus,expData(l,1:2),cell2mat(model(i,1))))');
    end
    
    for x=1:1000
        if(p(1) >= 0.001*x)        
            maxlab = 1;
        else
            maxlab = 2;
        end
        DETLabels(l,x) = maxlab;
    end
    
 end
 
    
 roc = zeros(1000,2);
 for x=1:1000  
      CM = zeros(2,k);
     for i=1:m  
        CM(DETLabels(i,x),expData(i,3)) = CM(DETLabels(i,x),expData(i,3)) + 1;    
    end
    roc(x,1) = log(CM(2,1)/sum(CM(:,1)));
    roc(x,2) = log(CM(1,2:k)/sum(CM(:,2:k)));
 end
 

    plot(roc(:,1),roc(:,2),'b-')
    hold on
    %plot([0 roc(1000,1)],[0 roc(1000,2)],'r-')
    %hold on
    %plot([1 roc(1,1)],[1 roc(1,2)],'r-')
    %xlim([-100,100])
    %ylim([-100,100])
    
 end
    