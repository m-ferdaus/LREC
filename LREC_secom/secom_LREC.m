clc
clear 

load dataset_secom_load

tic
CV=1;
CV2=1;
for j=1:CV2
for i=1:CV
Data=dataset;

ninput=size(dataset,2)-2;



[N1,N2]=size(Data);

 
b1=7*10^(-2);
b2=9*10^(-1);
c1=0.01;  
c2=0.01;  
parameters(1)=b1;
parameters(2)=b2;
parameters(3)=c1;
parameters(4)=c2;

fix_the_model=1096;
eta=1.0;
mode='c';
[y_palm,Weight,rule_palm,time_palm,classification_rate_training_palm,classification_rate_testing_palm,ConMAT,ConMAT_train,ConMAT_test]=LREC(Data,ninput,fix_the_model,parameters,eta);



Confusion_Matrix_all{j*i}=ConMAT;
ConMAT_train_all{j*i}=ConMAT_train;
ConMAT_test_all{j*i}=ConMAT_test;
CLR_training_all_palm(j,i)=classification_rate_training_palm;
CLR_testing_all_palm(j,i)=classification_rate_testing_palm;
rule_all_palm(j,i)=rule_palm(1,fix_the_model);
rule_plot_data_sing(j*i,:)=rule_palm(1,1:fix_the_model);
rule_plot_data={rule_plot_data_sing,'rule_plot_data_sing'};
time_all_palm(j,i)=time_palm;



end
Overall_CR_best_LREC=max(CLR_testing_all_palm)*100;
Overall_CR_avg_LREC=sum(CLR_testing_all_palm)/CV*100;  fprintf('Overall_CR_mean_LREC:%.2f\n', Overall_CR_avg_LREC)
Overall_CR_std_LREC=std(CLR_testing_all_palm)*100; fprintf('Overall_CR_std_LREC :%.2f\n', Overall_CR_std_LREC)
end

%%%===time== calcu=======
time_avg=sum(time_all_palm)/CV;
time_std=std(time_all_palm);

%rule--------
rule_avg=sum(rule_all_palm)/CV;
rule_std=std(rule_all_palm);

%%


%%%%========IF for training start=======================
for aaa=1:length(ConMAT_train_all)
clear row_sums;
clear class_acccu;
for kkk=1:length(ConMAT_train_all{1,aaa})
ConMAT_train_ele=ConMAT_train_all{1,aaa};

row_sums_tr(kkk)=sum(ConMAT_train_ele(kkk,:));
d_ele = diag(ConMAT_train_ele);
if row_sums_tr>0
class_acccu_tr(kkk)=d_ele(kkk)/row_sums_tr(kkk);
end

end
avg_accu_tr(aaa)=sum(class_acccu_tr)/length(ConMAT_train_ele);
IF_tr(aaa)=1-((length(ConMAT_test_all{1,aaa}))/sum(row_sums_tr))*min(row_sums_tr);
end
accu_f_avg_tr=sum(avg_accu_tr)/length(ConMAT_train_all);
accu_f_std_tr=std(avg_accu_tr);
accu_f_max_tr=max(avg_accu_tr);

IF_f_avg_tr=sum(IF_tr)/length(ConMAT_train_all);
IF_f_std_tr=std(IF_tr);
IF_f_max_tr=max(IF_tr);
%%%%========IF for training end=======================



%%%%========IF for testing start=======================
for aaa=1:length(ConMAT_test_all)
clear row_sums;
clear class_acccu;
for kkk=1:length(ConMAT_test_all{1,aaa})
ConMAT_test_ele=ConMAT_test_all{1,aaa};

row_sums(kkk)=sum(ConMAT_test_ele(kkk,:));
d_ele = diag(ConMAT_test_ele);
if row_sums>0
class_acccu(kkk)=d_ele(kkk)/row_sums(kkk);
end

end
avg_accu(aaa)=sum(class_acccu)/length(ConMAT_test_ele);
IF(aaa)=1-((length(ConMAT_test_all{1,aaa}))/sum(row_sums))*min(row_sums);
end
Average_CR_avg_LREC=sum(avg_accu)/length(ConMAT_test_all)*100;  fprintf('Average_CR_avg_LREC:%.2f\n', Average_CR_avg_LREC)
Average_CR_std_LREC=std(avg_accu)*100;  fprintf('Average_CR_std_LREC:%.2f\n', Average_CR_std_LREC)
Average_CR_max_LREC=max(avg_accu)*100;

IF_f_avg_ts=sum(IF)/length(ConMAT_test_all);
IF_f_std_ts=std(IF);
IF_f_max_ts=max(IF);
%%%%========IF for testing end=======================

