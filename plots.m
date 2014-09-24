%%
%function to plot decision boundaries for all cases for a given dataset
    %load data
    load 'ls_group3.mat';
    trainData = ls_group3; 
    set(gcf,'Name','Decision Boundaries');
    %Build models for all cases
    for j=4:5
    model = BuildBaysianModel(trainData, 0, j);
    
 %%
    %Plot Decision Boundary
    W = cell(2,2);
    w = cell(2,1);
    ww = zeros(2);
    
    for i=1:2
        W(i) = mat2cell(0.5 * inv(cell2mat(model(i,2))));
        w(i) = mat2cell(inv(cell2mat(model(i,2)))*cell2mat(model(i,1))');
        ww(i) = -0.5*cell2mat(model(i,1))*inv(cell2mat(model(i,2)))*cell2mat(model(i,1))' - 0.5*log(det(cell2mat(model(i,2))));
    end

    Wk = cell2mat(W(2)) - cell2mat(W(1));
    wk = cell2mat(w(2)) - cell2mat(w(1));
    wwk = ww(2) - ww(1);

    subplot(3,2,j); ezplot(@(x,y)decision(x,y,Wk,wk,wwk));

end