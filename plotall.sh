# $1 : sample_data
# $2 : directry
# $3 : method
mkdir $2
./svm $1 --cross --plot-all $2 $3
ls $2/*.dat | xargs -n 1 -P 16 -I % ./py3/plotdata.py % $1 --save %.png
rm $2/*.dat