%带动量的Delta规则
%单输入，单输出，p个隐藏层
clc
clear all
close all
disp('go');

%样本
x=0:0.5:10;
y=0.12*exp(-0.213*x)+0.54*exp(-0.17*x).*sin(-1.23*x);
n=length(x);%样本个数  
%初始化网络参数
p=9;%隐藏层个数  
W=rand(p,2);%初始化输入层权值,带阈值
% W=rand(p,1);%初始化输入层权值,不带阈值
% V=rand(1,p+1);%初始化输出层权值,带阈值
V=rand(1,p);%初始化输出层权值，不带阈值
epochs=10000;%最大步数 
Epsilon=0.001;%初始化控制精度 
alpla=0.11;%学习速率  
% u(p+1)=-1;%常数项
%开始
for steps=1:epochs  
    E=0;  
    for i=1:n %样本个数  
        xd=[x(i);-1];   
%         neto=0;  
        for j=1:p   
            neti(j)=W(j,1)*xd(1)+W(j,2)*xd(2);  
%             neti(j)=W(j,1)*x(i);
            u(j)=sigmoid(neti(j));%隐藏层激活函数
            uk(j)=V(j)*u(j);
        end
%         o(i)=sum(uk)-V(p+1);%输出层激活函数f(x)=x-b
         o(i)=sum(uk);%不带阈值
        e=(1/2)*(y(i)-o(i))^2;  
        E=e+E;
        Delta(i)=y(i)-o(i);%没有梯度
%         Delta(i)=(d(i)-o(i))*o(i)*(1-o(i));%有梯度
        delta_wk=alpla*Delta(i)*u;   
        for k=1:p  
            delta_w(k,1:2)=alpla*Delta(i)*V(k)*u(k)*(1-u(k))*xd; 
%             delta_w(k,1)=alpla*Delta(i)*V(k)*u(k)*(1-u(k))*x(i); 
        end     
        V=V+delta_wk; %从隐层到输出层权值的更新  
        W=W+delta_w; %从输入层到隐层的权值的更新      
    end   
    error(steps)=E;  
    m(steps)=steps;      
    if(E<Epsilon)              
       break;  
    elseif(steps==epochs)  
        disp('步数不够')          
    end   
end  
%泛化  
for i=1:n %样本个数  
    xd=[x(i);-1];    
%     neto=0;  
    for j=1:p  
        neti(j)=W(j,1)*xd(1)+W(j,2)*xd(2);  
%         neti(j)=W(j,1)*x(i);
        u(j)=sigmoid(neti(j));
        uk(j)=V(j)*u(j);
    end    
	o(i)=sum(uk);
end   
  
%误差图像  
figure(1)  
plot(m,error)  
xlabel('迭代次数')  
ylabel('均方误差')
%泛化图像
figure(2)  
plot(x,y)  
hold on  
plot(x,o,'r')  
legend('样本','泛化')  
disp('done')