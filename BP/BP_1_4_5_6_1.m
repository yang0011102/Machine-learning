%多输入，多输出，隐藏层3层，4,5,6
clc
clear all
close all
disp('go');
%样本
x=rand(20,2);
y=0.12*exp(-0.213*x)+0.54*exp(-0.17*x).*sin(-1.23*x);
num=size(x,2);
n=size(x,1);%样本个数  
%初始化网络
p=[4 5 6];

W1=rand(2,p(1));
W2=rand(p(1),p(2));
W3=rand(p(2),p(3));
W4=rand(p(3),2);
epochs=10000;
Epsilon=0.02;
alpha=0.1;
%
for steps=1:epochs
    E=0;
    for a=1:n
        xd=x(a,:);
        yd=y(a,:);
        u1i=xd*W1;
        u1=logsig(u1i);
        u2i=u1*W2;
        u2=logsig(u2i);
        u3i=u2*W3;
        u3=tansig(u3i);
        o=u3*W4;
        %反向传播 W4
        Delta_w4=yd-o;%误差矩阵No.4
            e=(Delta_w4.*Delta_w4)/2;
            E=sum(e);
        for i=1:num
            delta_w4(i,:)=alpha*Delta_w4(i).*u3;
        end
        W4=W4+delta_w4';
        %W3
        Delta_w3=(u3.*(1-u3))'.*(W4*Delta_w4');
        for i=1:p(3)
            delta_w3(i,:)=alpha*Delta_w3(i).*u2;
        end
        W3=W3+delta_w3';
        %W2
        Delta_w2=(u2.*(1-u2))'.*(W3*Delta_w3);
        for i=1:p(2)
            delta_w2(i,:)=alpha*Delta_w2(i).*u1;
        end
        W2=W2+delta_w2';
        %W1
        Delta_w1=(u1.*(1-u1))'.*(W2*Delta_w2);
        for i=1:p(1)
            delta_w1(i,:)=alpha*Delta_w1(i).*xd;
        end
        W1=W1+delta_w1';
    end
    error(steps)=E;  
    m(steps)=steps;      
    if(E<Epsilon)              
       break;
    elseif(steps==epochs)  
        disp('步数不够');
    end   
end
plot(error);
steps

disp('done');