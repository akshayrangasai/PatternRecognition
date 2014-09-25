%%

    load 'data_NLS_firstfile.mat';
    trainData = class_var; 
    m = size(trainData, 1); % number of training examples
    n = size(trainData, 2) - 1; % number of feature dimension
    k = length(unique(trainData(:, end))); % number of classes
    class_matrices = zeros(m,n,k);
    counts = ones(1,k);
    
    for i = 1:m
        class_matrices(counts(1,trainData(i,n+1)),:,trainData(i,n+1)) = trainData(i,1:n);
        counts(1,trainData(i,n+1)) = counts(1,trainData(i,n+1)) + 1;
    end

%function to plot decision boundaries for all cases for a given dataset
    %load data

    xmin = min(trainData(:,1));
    ymin = min(trainData(:,2));
    xmax = max(trainData(:,1));
    ymax = max(trainData(:,2));
 
    clf;
    set(gcf,'Name','Decision Boundaries'); 
    whitebg('black');
    %Build models for all cases
    for j=1:5
    model = BuildBaysianModel(trainData, 0, j);
    
    mean_1 = cell2mat(model(1,1));
    mean_2 = cell2mat(model(2,1));
    mean_3 = cell2mat(model(3,1));
    
    %%
    %Plot Decision Boundary
    W = cell(2,2);
    w = cell(2,1);
    ww = zeros(2);
    
    for i=1:k
        W(i) = mat2cell(-0.5 * inv(cell2mat(model(i,2))));
        w(i) = mat2cell(inv(cell2mat(model(i,2)))*cell2mat(model(i,1))');
        ww(i) = -0.5*cell2mat(model(i,1))*inv(cell2mat(model(i,2)))*cell2mat(model(i,1))' - 0.5*log(det(cell2mat(model(i,2))));
    end

    subplot(3,2,j);
    
    plot([mean_1(1,1) mean_2(1,1)], [mean_1(1,2) mean_2(1,2)],'white');
    hold on
    plot([mean_1(1,1) mean_3(1,1)], [mean_1(1,2) mean_3(1,2)],'white');
    hold on
    
    
    colorchart = ['r' 'g' 'b' 'c'];
    for i=1:k
        scatter(class_matrices(1:counts(1,trainData(1,n+1))-1,1,i),class_matrices(1:counts(1,trainData(1,n+1))-1,2,i), 5 ,colorchart(i));
        hold on
    end
    
    Wk = cell2mat(W(2)) - cell2mat(W(1));
    wk = cell2mat(w(2)) - cell2mat(w(1));
    wwk = ww(2) - ww(1);
  
    ezplot(@(x,y)decision(x,y,Wk,wk,wwk),[xmin, xmax, ymin, ymax]);
    setcurve ('color', 'yellow');
    hold on
    
    Wk = cell2mat(W(3)) - cell2mat(W(1));
    wk = cell2mat(w(3)) - cell2mat(w(1));
    wwk = ww(3) - ww(1);
  
    ezplot(@(x,y)decision(x,y,Wk,wk,wwk),[xmin, xmax, ymin, ymax]);
    setcurve ('color', 'magenta');
    hold on

end