% This function finds a linear discriminant using LP
% The linear discriminant is represented by 
% the weight vector w and the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [w,theta,delta] = findLinearDiscriminant(data)
%% setup linear program
[m, np1] = size(data);
n = np1-1;

% write your code here
x = data(:,1:n);
y = data(:,n+1);

c = zeros(n+2,1);
c(n+2) = 1;

b = ones(m+1,1);
b(m+1) = 0;

A = zeros(m+1,n+2);
for i = 1:m
    A(i,n+1) = y(i);
    A(i,n+2) = 1;
    A(i, 1:n) = y(i) * x(i, 1:n);
end
A(m+1,n+2) = 1;

%% solve the linear program
%adjust for matlab input: A*x <= b
[t, z] = linprog(c, -A, -b);

%% obtain w,theta,delta from t vector
w = t(1:n);
theta = t(n+1);
delta = t(n+2);

end
