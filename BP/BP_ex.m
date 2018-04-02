%function output = ANN_Training(X, t, H, max_iter)
%Train ANN with one hidden layer and one output unit for classification
%input :
%X:  attributes. Every column represents a sample.
%y:  target. should be 0 or 1.  length(y) == size(X,2) is assumed.
%H: size of hidden layer.
%max_iter: maximum iterates
%tol: convergence tolerate
%output:
% output: a structure containing all network parameters.
%Created by Ranch Y.Q. Lai on Mar 22, 2011
%ranchlai@163.com
 function output = ANN_Training(X, t, H, max_iter,tol)
[n,N] = size(X);
if N~= length(t)
    error('inconsistent sample size');
end
W = randn(n,H); % weight for hidden layer, W(:,i) is the weight vector for unit i
b = randn(H,1); % bias for hidden layer
wo = randn(H,1); %weight for output layer
bo = randn(1,1); %bias, output
y = zeros(H,1); %output of hidden layer
iter = 0;
cost_v = [inf];
fprintf('###################################\n');
fprintf('ANN TRAINING STARTED\n');
fprintf('###################################\n');
fprintf('Iterate \t training error\n');
while iter < max_iter
    delta_wo = zeros(H,1);
    delta_bo = 0;
    delta_W = zeros(n,H);
    delta_b = zeros(H,1);
    for i=1:N
        for j=1:H
            y(j) = s(X(:,i),W(:,j),b(j));
        end
        delta_wo = delta_wo + y*(s(y,wo,bo)-t(i))*s(y,wo,bo)*(1-s(y,wo,bo));
        delta_bo = delta_bo + (s(y,wo,bo)-t(i))*s(y,wo,bo)*(1-s(y,wo,bo));
        for j=1:H
            delta_W(:,j) = delta_W(:,j) + X(:,i)*(s(y,wo,bo)-t(i))*s(y,wo,bo)*(1-s(y,wo,bo))*wo(j)*s(X(:,i),W(:,j),b(j))*(1-s(X(:,i),W(:,j),b(j)));
            delta_b(j) = delta_b(j) + (s(y,wo,bo)-t(i))*s(y,wo,bo)*(1-s(y,wo,bo))*wo(j)*s(X(:,i),W(:,j),b(j))*(1-s(X(:,i),W(:,j),b(j)));            
        end
    end
    step = 1;
    cost =  training_error(N,H,X,t,W,b,wo,bo);
    while step > 1e-10
        wo1 = wo - step*delta_wo;
        bo1 = bo - step*delta_bo;
        W1 = W - step.*delta_W;
        b1 = b - step.*delta_b;
        cost1 =  training_error(N,H,X,t,W1,b1,wo1,bo1);
        if cost1 < cost
            break;
        end
        step = step * 0.1;
    end
    if step <=1e-10
        disp('cannot descend anymore');
        break;
    end
    % update wo,bo,W,b
    wo = wo1;
    bo = bo1;
    W = W1;
    b = b1;
    cost  = cost1;
    if iter>0
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
    end
    fprintf('%4d\t%1.10f\n',iter,cost);
    cost_v = [cost_v;cost];
    if abs(cost - cost_v(end-1))< tol
        break;
    end
    iter = iter+1;
end
output.W = W;
output.b= b;
output.wo = wo;
output.bo =bo;
output.cost = cost_v;
function cost = training_error(N,H,X,t,W,b,wo,bo)
%total cost
y = zeros(H,1);
cost = 0;
for i=1:N
    for j=1:H
        y(j) = s(X(:,i),W(:,j),b(j));
    end
    cost = cost + ( s(y,wo,bo) - t(i)) ^2;
end
cost = cost / N;