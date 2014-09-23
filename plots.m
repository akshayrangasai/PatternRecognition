function plots(model, data)

%decision boundary
    W= zeros(2,2,2);
    w = zeros(2,1,2);
    ww = zeros(2);
    for i=1:2
        W(i) = 0.5 * inv(cell2mat(model(i,2)));
        w(i) = inv(cell2mat(model(i,2)))*cell2mat(model(i,1));
        ww(i) = -0.5*cell2mat(model(i,1))'*inv(cell2mat(model(i,2))*cell2mat(model(i,1)) - 0.5*log(det(cell2mat(model(i,2))));
    end

    Wk = W(:,:,2) - W(:,:,1);
    wk = w(:,:,2) - w(:,:,1);
    wwk = ww(2) - ww(1);

    function z = decision(x,y,Wk,wk,wwk)
        z = (x^2)*Wk(1,1) + (y^2)*Wk(2,2) + x*y*(Wk(1,2)+Wk(2,1)) + x*wk(1,1) + y*wk(2,1) + wwk;
    end

    ezplot(@(x,y)decision(x,y,Wk,wk,wwk));

end

    

