%生成地图数据组
clc
clear all
close all
disp('go');

D=30;
num=20;
Q=cell(num,1);
W=cell(num,1);
E=cell(num,1);
for c=1:num
    [A,d,r]=randompic(D);
    Q{c,1}=A;
    W{c,1}=d;
    E{c,1}=r;
end
T=[Q;W];
disp('done');