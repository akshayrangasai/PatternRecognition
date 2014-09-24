%%decision boundary function
function z = decision(x,y,Wk,wk,wwk)
    z = (x^2)*Wk(1,1) + (y^2)*Wk(2,2) + x*y*(Wk(1,2)+Wk(2,1)) + x*wk(1,1) + y*wk(2,1) + wwk;
end