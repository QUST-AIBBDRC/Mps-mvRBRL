clear all
clc
lambda=49;
%%%%%�ҳ����ݼ�������
%��str=
%%%%%    
fid=fopen('yangxingjun');
string=fscanf(fid,'%s'); %�ļ�����
%ƥ����ַ���
firstmatches=findstr(string,'>')+7;%��ʼλ��
endmatches=findstr(string,'>')-1;
firstnum=length(firstmatches); %firstnum=endnum���е�����
endnum=length(endmatches);
  for k=1:firstnum-1
    j=1;
    lensec(k)=endmatches(k+1)-firstmatches(k)+1;%ÿ�����еĳ���
   for mm=firstmatches(k):endmatches(k+1)
        sequence(k,j)=string(mm); %�ַ�����
        j=j+1;
   end
   
  end
for i=1:firstnum-1
paac(i,:)= PAAC(sequence(i,1:lensec(i)),lambda);
end
save  xinne-pseaac49.mat paac


