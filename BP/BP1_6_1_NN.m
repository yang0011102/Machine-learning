%��������Delta����
%�����룬�������p�����ز�
clc
clear all
close all
disp('go');

%����
x=0:0.5:10;
y=0.12*exp(-0.213*x)+0.54*exp(-0.17*x).*sin(-1.23*x);
n=length(x);%��������  
%��ʼ���������
p=9;%���ز����  
W=rand(p,2);%��ʼ�������Ȩֵ,����ֵ
% W=rand(p,1);%��ʼ�������Ȩֵ,������ֵ
% V=rand(1,p+1);%��ʼ�������Ȩֵ,����ֵ
V=rand(1,p);%��ʼ�������Ȩֵ��������ֵ
epochs=10000;%����� 
Epsilon=0.001;%��ʼ�����ƾ��� 
alpla=0.11;%ѧϰ����  
% u(p+1)=-1;%������
%��ʼ
for steps=1:epochs  
    E=0;  
    for i=1:n %��������  
        xd=[x(i);-1];   
%         neto=0;  
        for j=1:p   
            neti(j)=W(j,1)*xd(1)+W(j,2)*xd(2);  
%             neti(j)=W(j,1)*x(i);
            u(j)=sigmoid(neti(j));%���ز㼤���
            uk(j)=V(j)*u(j);
        end
%         o(i)=sum(uk)-V(p+1);%����㼤���f(x)=x-b
         o(i)=sum(uk);%������ֵ
        e=(1/2)*(y(i)-o(i))^2;  
        E=e+E;
        Delta(i)=y(i)-o(i);%û���ݶ�
%         Delta(i)=(d(i)-o(i))*o(i)*(1-o(i));%���ݶ�
        delta_wk=alpla*Delta(i)*u;   
        for k=1:p  
            delta_w(k,1:2)=alpla*Delta(i)*V(k)*u(k)*(1-u(k))*xd; 
%             delta_w(k,1)=alpla*Delta(i)*V(k)*u(k)*(1-u(k))*x(i); 
        end     
        V=V+delta_wk; %�����㵽�����Ȩֵ�ĸ���  
        W=W+delta_w; %������㵽�����Ȩֵ�ĸ���      
    end   
    error(steps)=E;  
    m(steps)=steps;      
    if(E<Epsilon)              
       break;  
    elseif(steps==epochs)  
        disp('��������')          
    end   
end  
%����  
for i=1:n %��������  
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
  
%���ͼ��  
figure(1)  
plot(m,error)  
xlabel('��������')  
ylabel('�������')
%����ͼ��
figure(2)  
plot(x,y)  
hold on  
plot(x,o,'r')  
legend('����','����')  
disp('done')