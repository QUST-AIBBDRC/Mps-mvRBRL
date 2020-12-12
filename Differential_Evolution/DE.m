
clear
clc
%��ʼ��
maxIteration=100;%����������
Generation=1;%�������������ߵ�ǰ��������
Xmax=2;%�����Ͻ磬���Ը�����Ҫ��Ϊ������ʽ��Ȩ�ص����½�
Xmin=-2;%�����½�
Dim=5;%����ά��������������ȡ�����ĸ���
NP=10;%population size,��Ⱥ��ģ
F=0.5;%scaling factor ��������
CR=0.3;%crossover rate �������

mutationStrategy=1;%�������
crossStrategy=1;%�������
%%
%step1 ��ʼ�������ɵ�һ���ľ���
%X represent population
%Generation=0;
X=(Xmax-Xmin)*rand(NP,Dim)+Xmin;%X�д������i���д������i��ά��j
 
%%
%step2 mutation,crossover,selection   ѭ������
while Generation<maxIteration  %��������һֱѭ����ȥ
%��bestX
    for i=1:NP 
        fitnessX(i)=testFun(X(i,:));
    end
    [fitnessbestX,indexbestX]=max(fitnessX); %��ǰ��������ѵ�Ȩ�ز�����������ֹͣʱ����������յĽ��
    bestX=X(indexbestX,:);   %bestX��ʾ����ֵ
%%
%step2.1 mutation ͻ�䣬���������ж�Ӧ��m��ͻ��Ĳ��Կ���ѡ��������ʹ�ò���1
%mutationStrategy=1��DE/rand/1,
%mutationStrategy=2��DE/best/1,
%mutationStrategy=3��DE/rand-to-best/1,
%mutationStrategy=4��DE/best/2,
%mutationStrategy=5��DE/rand/2,
%����Ϊÿһ������Xi,G ����һ����������Vi,G�� G�����������
    V=mutation(X,bestX,F,mutationStrategy); %V����ͻ�������Ⱥ��
 %%   
%step2.2 crossover,����
%crossStrategy=1:binomial crossover
%crossStrategy=2:Exponential crossover
%����Ϊÿһ������Xi,G ����һ����������Ui,G�� G�����������

    U=crossover(X,V,CR,crossStrategy);
%%    
%step2.3 selection ѡ����ͨ��������������ӵĺ�������ѡ��ģ��ж��µ�U�Ƿ��X��
    for i=1:NP
        fitnessU(i)=testFun(U(i,:));  %��Ӧ����ָ���ֵ
        if fitnessU(i)>=fitnessX(i)
            X(i,:)=U(i,:);
            fitnessX(i)=fitnessU(i);
            if fitnessU(i)>fitnessbestX
                bestX=U(i,:);
                fitnessbestX=fitnessU(i);
            end
        end
    end
%%
    Generation=Generation+1;
    bestfitnessG(Generation)=fitnessbestX;
end
 %%%%����bestX����ѵ�Ȩ�ر�ʾ��fitnessbestX�Ƕ�Ӧ�����Ȩ�ر�ʾ������ָ���ֵ
%%
% %��ͼ
plot(bestfitnessG);
optValue=num2str(fitnessbestX);
Location=num2str(bestX);
disp(strcat('the optimal value','=',optValue));
disp(strcat('the best location','=',Location));
 
 
