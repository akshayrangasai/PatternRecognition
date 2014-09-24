%%
%function to plot decision boundaries for all cases for a given dataset
    %load data
    load 'ls_group3.mat';
    trainData = ls_group3; 
    xmin = min(trainData(:,1));
    ymin = min(trainData(:,2));
    xmax = max(trainData(:,1));
    ymax = max(trainData(:,2));
 
    set(gcf,'Name','Decision Boundaries'); 
    %Build models for all cases
    for j=1:5
    model = BuildBaysianModel(trainData, 0, j);
    mean_1 = cell2mat(model(1,1));
    mean_2 = cell2mat(model(2,1));
 %%
    %Plot Decision Boundary
    W = cell(2,2);
    w = cell(2,1);
    ww = zeros(2);
    
    for i=1:2
        W(i) = mat2cell(-0.5 * inv(cell2mat(model(i,2))));
        w(i) = mat2cell(inv(cell2mat(model(i,2)))*cell2mat(model(i,1))');
        ww(i) = -0.5*cell2mat(model(i,1))*inv(cell2mat(model(i,2)))*cell2mat(model(i,1))' - 0.5*log(det(cell2mat(model(i,2))));
    end

    Wk = cell2mat(W(2)) - cell2mat(W(1));
    wk = cell2mat(w(2)) - cell2mat(w(1));
    wwk = ww(2) - ww(1);

    subplot(3,2,j); 
    scatter(trainData(1:500,1),trainData(1:500,2),'red')
    hold on
    scatter(trainData(501:1000,1),trainData(501:1000,2),'green')
    hold on
%    scatter(trainData(1001:1500,1),trainData(1001:1500,2),'blue')
%    hold on
%    scatter(trainData(1501:2000,1),trainData(1501:2000,2),'black')
%    hold on
    ezplot(@(x,y)decision(x,y,Wk,wk,wwk),[xmin, xmax, ymin, ymax]);
    setcurve ('color', 'black');
    hold on
    plot([mean_1(1,1) mean_2(1,1)], [mean_1(1,2) mean_2(1,2)],'yellow');

end