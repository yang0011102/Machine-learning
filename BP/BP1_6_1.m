%��������Delta����
clc
clear all
close all
disp('go');

for i=1:20 %��������  
    xx(i)=2*pi*(i-1)/20;  
    d(i)=0.5*(1+cos(xx(i)));  
end  
n=length(xx);%��������  
p=6; %�������  
w=rand(p,2);  
wk=rand(1,p+1);  
max_epoch=10000;%���ѵ������  
error_goal=0.002;%�������  
q=0.09;%ѧϰ����  
a(p+1)=-1;  
  
%training  
%��ѵ�������ȡ1-6-1����ʽ����һ�����룬6�����㣬1�����  
for epoch=1:max_epoch  
    e=0;  
    for i=1:n %��������  
        x=[xx(i);-1];   
        neto=0;  
        for j=1:p   
            neti(j)=w(j,1)*x(1)+w(j,2)*x(2);  
            a(j)=1/(1+exp(-neti(j)));%����ļ������ȡs������f(x)=1/(1+exp(-x))  
            neto=neto+wk(j)*a(j);  
        end
        neto=neto+wk(p+1)*(-1);  
        y(i)=neto; %�����ļ������ȡ���Ժ���,f(x)=x  
        de=(1/2)*(d(i)-y(i))*(d(i)-y(i));  
        e=de+e;       
        delta_wk=q*(d(i)-y(i))*a;   
        for k=1:p  
            delta_w(k,1:2)=q*(d(i)-y(i))*wk(k)*a(k)*(1-a(k))*x;      
%             delta_w(k,1:2)=q*wk(k)*a(k)*(1-a(k))*x;     
        end     
        wk=wk+delta_wk; %�����㵽�����Ȩֵ�ĸ���  
        w=w+delta_w; %������㵽�����Ȩֵ�ĸ���      
    end   
    error(epoch)=e;  
    m(epoch)=epoch;      
    if(e<error_goal)              
       break;  
    elseif(epoch==max_epoch)  
        disp('��Ŀǰ�ĵ��������ڲ��ܱƽ�������������Ӵ��������')          
    end   
end  
%simulation  
for i=1:n %��������  
    x=[xx(i);-1];    
    neto=0;  
    for j=1:p  
        neti(j)=w(j,1)*x(1)+w(j,2)*x(2);  
        a(j)=1/(1+exp(-neti(j)));  
        neto=neto+wk(j)*a(j);  
    end    
    neto=neto+wk(p+1)*(-1);  
    y(i)=neto; %���Ժ���  
end   
  
%plot  
figure(1)  
plot(m,error)  
xlabel('��������')  
ylabel('�������')  
title('BP�㷨��ѧϰ����')  
figure(2)  
plot(xx,d)  
hold on  
plot(xx,y,'r')  
legend('������Ŀ������','�����Ǳƽ�����')  
disp('done')