WEISHU=523;
load('nexuliechangdu.mat')
for i=1:WEISHU
    nnn=num2str(i);
    name = strcat(nnn,'.pssm');
    fid{i}=importdata(name);
end
C={};
for t=1:WEISHU
    shu=fid{t}.data;
    shuju=shu(1:len(1,t),1:20);
    C{t}=shuju;
end
 save shuju.mat C   
 %��ÿ��pssm�ļ������һ��cell���ŵ�.mat�ļ��¡�
 


