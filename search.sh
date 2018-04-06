echo "use gpu with multiple processes";
#list="SST_2 SST_5 MPQA TREC SUBJ MR CR"
list="SST_5 MPQA TREC SUBJ MR CR"
count = 7
for dataset in $list
do
    echo process data $dataset 
    for((i=0;i<$count;i++))
    do
        {
         echo "use gpu" +$i ; 
         echo CUDA_VISIBLE_DEVICES=$i python multi_search.py -gpu $i -dataset $dataset -count $count; 
         CUDA_VISIBLE_DEVICES=$i python multi_search.py -gpu $i -dataset $dataset -count $count; 
         
         }&
    done
    wait
    echo finished with dataset: $dataset 
done