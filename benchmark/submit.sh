method="krpa"
mkdir -p $method; cd $method

for cell in diamond; do
    mkdir -p $cell

    cd $cell; cp ../../submit.py .
    python submit.py --time 00:10:00 --cpus-per-task 32

    cd ..
done
