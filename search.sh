echo "use gpu with multiple processes";
for((i=1;i<=8;i++))
do
    {
     echo "use gpu" +$i ; CUDA_VISIBLE_DEVICES=$i python multi_search.py -gpu $i 
     
     }
done