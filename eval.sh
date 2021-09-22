n=
c=0
for i in `seq 0 30`
do
        s=$((i * 38))
        e=$(((i + 1) * 38))
        echo "conversation no =  $i ($s ~ $e)"
        nice python main.py $s $e &
done
wait
