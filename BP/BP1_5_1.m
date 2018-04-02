clc
clear all
close all
disp('GO');
%样本
x=0:0.5:10;
y=0.12*exp(-0.213*x)+0.54*exp(-0.17*x).*sin(-1.23*x);
%BP网络
P=[1,5,1];%网络结构
h=length(P);%总层数

%Delta规则
alpha=0.05;%学习速率，0.01~0.1

W=rand(P(1,2),1);%初始化输出层权值
V=rand(1,P(1,2));%初始化隐藏层权值
Epsilon=0.1;%初始化控制精度
E=Epsilon+1;

% while E>Epsilon
    E=0;%初始化误差
%     for num_x=1:length(x)
%        num=3;
       for i=1:P(1,2)
           u(i,1)=x(num_x)*V(i);
           uk(i,1)=logsig(u(i));%激活函数1
           ok(i,1)=uk(i)*W(i);
           O(num_x,1)=tansig(sum(ok));%激活函数2
       end
%     end
%     for num_y=1:length(y)
        Ep(num_y,1)=((O(num_y)-y(num_y))^2)/2;
%     end
    E=sum(Ep);
    
    for j=1:P(1,2)%更新输出层权值
        delta(j,1)=(1-O(1,P))*(y(1,P)-O(1,P));%单层误差
        delta_W(j,1)=alpha*delta(j,1)*uk(j,1);
        W(j,1)=W(j,1)+delta_W(j,1);
    end
% 	for j=1:h%更新隐藏层权值
%         delta_V(j,1)=alpha*;
%         W(j,1)=W(j,1)+delta_W(j,1);
%     end
% end

disp('DONE');