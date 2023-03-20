for i in {1..9}
do
    for j in {1..10}
    do
        ./con <in$i >out.txt 2>/dev/null
        python3 measure.py $j
    done
done