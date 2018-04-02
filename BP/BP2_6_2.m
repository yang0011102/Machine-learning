%��������Delta����
%2���룬2�����p�����ز�
clc
clear all
close all
disp('go');

%����
x=rand(20,2);
y=0.12*exp(-0.213*x)+0.54*exp(-0.17*x).*sin(-1.23*x);
num=size(x,2);
n=size(x,1);%��������  
%��ʼ������
p=6;
W=rand(p,2);
V=rand(p,2);
epochs=1000;
Epsilon=0.02;
alpha=0.1;

%��ʼ
for steps=1:epochs
    E=0;
    for a=1:n%������������
        xd=x(a,:);
        yd=y(a,:);
        neti=xd*W';
        u=logsig(neti);
        uk=u*V;
        o=uk;
        Delta=yd-o;
        e=(Delta.*Delta)/2;
        E=E+sum(e);
        for i=1:num
            delta_v(i,:)=alpha*Delta(i).*u';
        end        
        for i=1:num
            delta_w(i,:)=alpha*(u.*(1-u)).*(xd*Delta(i)*V');
        end
        V=V+delta_v';
        W=W+delta_w';
    end
    error(steps)=E;  
    m(steps)=steps;      
    if(E<Epsilon)              
       break;
    elseif(steps==epochs)  
        disp('��������');
    end   
end
figure(1)
plot(error);
disp('done')