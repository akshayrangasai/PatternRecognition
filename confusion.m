    load 'ls_group3.mat';
    trainData = [ls_group3(1:400,:); ls_group3(501:900,:); ls_group3(1001:1400,:); ls_group3(1501:1900,:)];
    expData = [ls_group3(401:500,:); ls_group3(901:1000,:); ls_group3(1401:1500,:); ls_group3(1901:2000,:)];
    k = length(unique(trainData(:, end)));
    
    for z = 1:5
    model = BuildBaysianModel(trainData,0,z); %change case here
    expResults = BayesianClassify(model,expData(:,1:2));
    
    CM = zeros(k,k);
    
    for i=1:400
        CM(expResults(i),expData(i,3)) = CM(expResults(i),expData(i,3)) + 1;
    end
    
    CM
    end
        
    