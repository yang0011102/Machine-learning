%带动量的Delta规则
clc
clear all
close all
disp('go');

for i=1:20 %样本个数  
    xx(i)=2*pi*(i-1)/20;  
    d(i)=0.5*(1+cos(xx(i)));  
end  
n=length(xx);%样本个数  
p=6; %隐层个数  
w=rand(p,2);  
wk=rand(1,p+1);  
max_epoch=10000;%最大训练次数  
error_goal=0.002;%均方误差  
q=0.09;%学习速率  
a(p+1)=-1;  
  
%training  
%此训练网络采取1-6-1的形式，即一个输入，6个隐层，1个输出  
for epoch=1:max_epoch  
    e=0;  
    for i=1:n %样本个数  
        x=[xx(i);-1];   
        neto=0;  
        for j=1:p   
            neti(j)=w(j,1)*x(1)+w(j,2)*x(2);  
            a(j)=1/(1+exp(-neti(j)));%隐层的激活函数采取s函数，f(x)=1/(1+exp(-x))  
            neto=neto+wk(j)*a(j);  
        end
        neto=neto+wk(p+1)*(-1);  
        y(i)=neto; %输出层的激活函数采取线性函数,f(x)=x  
        de=(1/2)*(d(i)-y(i))*(d(i)-y(i));  
        e=de+e;       
        delta_wk=q*(d(i)-y(i))*a;   
        for k=1:p  
            delta_w(k,1:2)=q*(d(i)-y(i))*wk(k)*a(k)*(1-a(k))*x;      
%             delta_w(k,1:2)=q*wk(k)*a(k)*(1-a(k))*x;     
        end     
        wk=wk+delta_wk; %从隐层到输出层权值的更新  
        w=w+delta_w; %从输入层到隐层的权值的更新      
    end   
    error(epoch)=e;  
    m(epoch)=epoch;      
    if(e<error_goal)              
       break;  
    elseif(epoch==max_epoch)  
        disp('在目前的迭代次数内不能逼近所给函数，请加大迭代次数')          
    end   
end  
%simulation  
for i=1:n %样本个数  
    x=[xx(i);-1];    
    neto=0;  
    for j=1:p  
        neti(j)=w(j,1)*x(1)+w(j,2)*x(2);  
        a(j)=1/(1+exp(-neti(j)));  
        neto=neto+wk(j)*a(j);  
    end    
    neto=neto+wk(p+1)*(-1);  
    y(i)=neto; %线性函数  
end   
  
%plot  
figure(1)  
plot(m,error)  
xlabel('迭代次数')  
ylabel('均方误差')  
title('BP算法的学习曲线')  
figure(2)  
plot(xx,d)  
hold on  
plot(xx,y,'r')  
legend('蓝线是目标曲线','红线是逼近曲线')  
disp('done')