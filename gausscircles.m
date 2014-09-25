    load 'ls_group3.mat';
    trainData = ls_group3; 
   
    m = size(trainData, 1); % number of training examples
    n = size(trainData, 2) - 1; % number of feature dimension
    k = length(unique(trainData(:, end))); % number of classes
    class_matrices = zeros(m,n,k);
    counts = ones(1,k);
    
    for i = 1:m
        class_matrices(counts(1,trainData(i,n+1)),:,trainData(i,n+1)) = trainData(i,1:n);
        counts(1,trainData(i,n+1)) = counts(1,trainData(i,n+1)) + 1;
    end

    set(gcf,'Name','Decision Boundaries'); 
    colorchart = ['r' 'g' 'b' 'c'];

    for i=1:5
       
        subplot(3,2,j);
     
        for j=1:k
        scatter(class_matrices(1:counts(1,trainData(1,n+1))-1,1,j),class_matrices(1:counts(1,trainData(1,n+1))-1,2,j), 5 ,colorchart(j));
        hold on
        end
    
        model = BuildBaysianModel(trainData,0,i);
        for j=1:k
           for p= 0.2:1
               ezplot(@(x1, x2)multigauss(x1,x2,k,cell2mat(model(i,1)),cell2mat(model(i,2)),p));
               hold on;
               setcurve ('color', 'cyan');
               p = p+0.2;
           end
       end
    end
    