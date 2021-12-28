clc
clear

load electricity_pricing.mat

nFolds=91;
ninput=8;
noutput=2;

sourcedata=processeddataset;
[nData,nData1]=size(sourcedata);
[creditcardoutput,pendigits_Data]=modify_dataset_zero_class(sourcedata);
chunk_size=floor(nData/nFolds);
data=[sourcedata(:,1:end-1) creditcardoutput];
network=[];
counter=0;
iii=0;
for  k=1:chunk_size:nData
    
    iii=iii+1;
    
    lambdaD=min(1-exp(-counter/nFolds),0.01);
    lambdaW=min(1-exp(-counter/(nFolds-1)),0.1);
    confidenceinterval=lambdaD;
    if (k+chunk_size-1) > nData
        Data = data(k:nData,:);
        Data2 = data(k:nData,:);
        
    else
        Data = data(k:(k+chunk_size-1),:);
        if ((iii+1)*(chunk_size)) > nData
            Data2 = data(chunk_size+k:nData,:);
        else
            Data2 = data(chunk_size+k:(iii+1)*(chunk_size),:);
        end
    end
    
    Dataa=[Data; Data2];
    [r,q]=size(Data);
    [upperbound,upperboundlocation]=max(Data(:,1:ninput));
    [lowerbound,lowerboundlocation]=min(Data(:,1:ninput));
    
    
    b1=10*10^(0);
    b2=2*10^(-1);
    c1=0.01;
    c2=0.01;
    parameters(1)=b1;
    parameters(2)=b2;
    parameters(3)=c1;
    parameters(4)=c2;
    
    fix_the_model=r;
    
    
    eta=1;
    mode='c';
    if iii==1
        [y_palm,Weight,rule_palm,time_palm,classification_rate_training_palm,classification_rate_testing_palm,ConMAT,ConMAT_train,ConMAT_test]=LREC(Dataa,ninput,fix_the_model,parameters,eta);
        
        nRule=rule_palm(1,fix_the_model);
    else
        network=struct('nRule',nRule,'Weight',Weight);
        
        [y_palm,Weight,rule_palm,time_palm,classification_rate_training_palm,classification_rate_testing_palm,ConMAT,ConMAT_train,ConMAT_test]=LREC2(Dataa,ninput,fix_the_model,parameters,eta,network.Weight,network.nRule);
        nRule=rule_palm(1,fix_the_model);
        network=struct('nRule',nRule,'Weight',Weight);
    end
    
    All_rule(iii)=nRule;
    CLR_training_all_palm(iii)=classification_rate_training_palm;
    CLR_testing_all_palm(iii)=classification_rate_testing_palm;
    ConMAT_test_all{iii}=ConMAT_test;
    
        TP = ConMAT_test(1,1);
    FP = ConMAT_test(1,2);
    TN = ConMAT_test(2,2);
    FN = ConMAT_test(2,1);
    
    Precision = TP / (TP+FP);
    Recall = TP / (TP+FN);
    
    P_all(iii)=Precision;
    R_all(iii)=Recall;
    
    
end

Overall_CR_best_LREC=max(CLR_testing_all_palm)*100;
Overall_CR_avg_LREC=sum(CLR_testing_all_palm)/nFolds*100;  fprintf('Overall_CR_mean_LREC:%.2f\n', Overall_CR_avg_LREC)
Overall_CR_std_LREC=std(CLR_testing_all_palm)*100; fprintf('Overall_CR_std_LREC :%.2f\n', Overall_CR_std_LREC)


rule_avg_LREC=sum(All_rule)/nFolds;  fprintf('rule_avg_LREC:%.2f\n', rule_avg_LREC)
rule_std_LREC=std(All_rule); fprintf('rule_std_LREC :%.2f\n', rule_std_LREC)

Pr_avg_LREC=sum(P_all)/nFolds;  fprintf('Pr_avg_LREC:%.2f\n', Pr_avg_LREC)
Pr_std_LREC=std(P_all); 

Rc_avg_LREC=sum(R_all)/nFolds;  fprintf('Rc_avg_LREC:%.2f\n', Rc_avg_LREC)
Rc_std_LREC=std(R_all); 
