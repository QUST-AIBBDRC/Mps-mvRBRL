
clear
clc
%初始化
maxIteration=100;%最大迭代次数
Generation=1;%进化代数，或者当前迭代代数
Xmax=2;%搜索上界，可以根据需要改为向量形式，权重的上下界
Xmin=-2;%搜索下界
Dim=5;%个体维数，代表特征提取方法的个数
NP=10;%population size,种群规模
F=0.5;%scaling factor 缩放因子
CR=0.3;%crossover rate 交叉概率

mutationStrategy=1;%变异策略
crossStrategy=1;%交叉策略
%%
%step1 初始化，生成第一代的矩阵
%X represent population
%Generation=0;
X=(Xmax-Xmin)*rand(NP,Dim)+Xmin;%X行代表个体i，列代表个体i的维度j
 
%%
%step2 mutation,crossover,selection   循环操作
while Generation<maxIteration  %满足条件一直循环下去
%求bestX
    for i=1:NP 
        fitnessX(i)=testFun(X(i,:));
    end
    [fitnessbestX,indexbestX]=max(fitnessX); %当前代数下最佳的权重参数，最后迭代停止时，这就是最终的结果
    bestX=X(indexbestX,:);   %bestX表示最优值
%%
%step2.1 mutation 突变，生成论文中对应的m，突变的策略可以选择，论文中使用策略1
%mutationStrategy=1：DE/rand/1,
%mutationStrategy=2：DE/best/1,
%mutationStrategy=3：DE/rand-to-best/1,
%mutationStrategy=4：DE/best/2,
%mutationStrategy=5：DE/rand/2,
%产生为每一个个体Xi,G 产生一个变异向量Vi,G。 G代表进化代数
    V=mutation(X,bestX,F,mutationStrategy); %V代表突变后整个群体
 %%   
%step2.2 crossover,交叉
%crossStrategy=1:binomial crossover
%crossStrategy=2:Exponential crossover
%产生为每一个个体Xi,G 产生一个交叉向量Ui,G。 G代表进化代数

    U=crossover(X,V,CR,crossStrategy);
%%    
%step2.3 selection 选择是通过上面我让你添加的函数进行选择的，判断新的U是否比X好
    for i=1:NP
        fitnessU(i)=testFun(U(i,:));  %对应评价指标的值
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
 %%%%最后的bestX是最佳的权重表示，fitnessbestX是对应的最佳权重表示的评价指标的值
%%
% %画图
plot(bestfitnessG);
optValue=num2str(fitnessbestX);
Location=num2str(bestX);
disp(strcat('the optimal value','=',optValue));
disp(strcat('the best location','=',Location));
 
 
