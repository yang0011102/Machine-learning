%生成随机地图
clc
clear all 
close all
D=20;
At=rand(D);
A=tril(At,-1)+triu(At',0);%邻接

for i=1:D
    for j=1:D
        if A(i,j)>0.8
            A(i,j)=1;
        else
            A(i,j)=0;
        end
    end
end
[p1,p2]=find(A==1);
X=unidrnd(100,D,1);
Y=unidrnd(100,D,1);
P=[X,Y];
L=inf*ones(D,D);
for i=1:size(p1,1)
    a=P(p1(i),:);
    b=P(p2(i),:);
    plot([a(1),b(1)],[a(2),b(2)],'k');
    L(p1(i),p2(i))=sqrt((a(1)-b(1))^2+(a(2)-b(2))^2);
    L(p2(i),p1(i))=sqrt((a(1)-b(1))^2+(a(2)-b(2))^2);
    hold on
end
for i=1:D
    L(i,i)=0;
end
[d,r]=floyd(L);
% plot(X,Y,'o');