clc
clear all
close all
disp('GO');
%����
x=0:0.5:10;
y=0.12*exp(-0.213*x)+0.54*exp(-0.17*x).*sin(-1.23*x);
%BP����
P=[1,5,1];%����ṹ
h=length(P);%�ܲ���

%Delta����
alpha=0.05;%ѧϰ���ʣ�0.01~0.1

W=rand(P(1,2),1);%��ʼ�������Ȩֵ
V=rand(1,P(1,2));%��ʼ�����ز�Ȩֵ
Epsilon=0.1;%��ʼ�����ƾ���
E=Epsilon+1;

% while E>Epsilon
    E=0;%��ʼ�����
%     for num_x=1:length(x)
%        num=3;
       for i=1:P(1,2)
           u(i,1)=x(num_x)*V(i);
           uk(i,1)=logsig(u(i));%�����1
           ok(i,1)=uk(i)*W(i);
           O(num_x,1)=tansig(sum(ok));%�����2
       end
%     end
%     for num_y=1:length(y)
        Ep(num_y,1)=((O(num_y)-y(num_y))^2)/2;
%     end
    E=sum(Ep);
    
    for j=1:P(1,2)%���������Ȩֵ
        delta(j,1)=(1-O(1,P))*(y(1,P)-O(1,P));%�������
        delta_W(j,1)=alpha*delta(j,1)*uk(j,1);
        W(j,1)=W(j,1)+delta_W(j,1);
    end
% 	for j=1:h%�������ز�Ȩֵ
%         delta_V(j,1)=alpha*;
%         W(j,1)=W(j,1)+delta_W(j,1);
%     end
% end

disp('DONE');