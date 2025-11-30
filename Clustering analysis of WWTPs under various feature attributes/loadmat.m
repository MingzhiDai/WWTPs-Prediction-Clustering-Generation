clear;
clc;

% filename='./seeds/';
% filename='./BalanceScale/';
% filename='./glass/';
% filename='./HagermansSurvival/';
% filename='./htru2/';
% filename='./User_knowledge/';
% filename='./vowel/';
 filename='./wine/';
data=xlsread(strcat(filename,'data.xlsx'));
datalabel=xlsread(strcat(filename,'datalabel.xlsx'));

savepath1=strcat(filename,'data.mat');
savepath2=strcat(filename,'datalabel.mat');
save(savepath1,'data');
save(savepath2,'datalabel');
a='done!'