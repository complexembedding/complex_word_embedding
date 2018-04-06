echo "use gpu with multiple processes";
#list="SST_2 SST_5 MPQA TREC SUBJ MR CR"
list="SST_5 MPQA TREC SUBJ MR CR"
for dataset in $list
do
    echo process data $dataset 
    for((i=0;i<=7;i++))
    do
        {
         echo "use gpu" +$i ; 
         echo CUDA_VISIBLE_DEVICES=$i python multi_search.py -gpu $i -dataset $dataset; 
         CUDA_VISIBLE_DEVICES=$i python multi_search.py -gpu $i -dataset $dataset; 
         
         }&
    done
    wait
    echo finished with dataset: $dataset 
done