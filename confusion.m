    load 'data_NLS_firstfile.mat';
    trainData = [class_var(1:400,:); class_var(501:900,:); class_var(1001:1400,:)];%; class_var(1501:1900,:)];
    expData = [class_var(401:500,:); class_var(901:1000,:); class_var(1401:1500,:)];%; class_var(1901:2000,:)];
    k = length(unique(trainData(:, end)));
    
    for z = 1:5
    model = BuildBaysianModel(trainData,0,z); %change case here
    expResults = BayesianClassify(model,expData(:,1:2));
    
    CM = zeros(k,k);
    
    for i=1:300
        CM(expResults(i),expData(i,3)) = CM(expResults(i),expData(i,3)) + 1;
    end
    
    CM
    end
        
    