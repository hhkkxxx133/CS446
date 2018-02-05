% This function solves the LP problem for a given weight vector
% to find the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [theta,delta] = findLinearThreshold(data,w)
%% setup linear program
[m, np1] = size(data);
n = np1-1;

% write your code here
x = data(:,1:n);
y = data(:,n+1);

c = [0;1];

A = ones(m+1,2);
for i=1:m
    A(i,1) = y(i);
end
A(m+1,1) = 0;

b = zeros(m+1,1);
for i=1:m
    b(i) = 1-y(i)*dot(w,x(i,:));
end    
b(m+1) = 0;

%% solve the linear program
%adjust for matlab input: A*x <= b
[t, z] = linprog(c, -A, -b);%, [], [], [w' -inf -inf], [w' inf inf]);

theta = t(1);
delta = t(2);

% %% obtain w,theta,delta from t vector
% w = t(1:n);
% theta = t(n+1);
% delta = t(n+2);

end
