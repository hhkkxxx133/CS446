% This function plots the linear discriminant.
% YOU NEED TO IMPLEMENT THIS FUNCTION

function plot2dSeparator(w, theta)
    x = linspace(-2,2,100);
    y = -w(1)/w(2)*x - theta/w(2);
    plot(x,y);
end
