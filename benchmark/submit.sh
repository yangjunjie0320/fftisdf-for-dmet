method="kmesh"
mkdir -p $method; cd $method

for cell in diamond; do
    mkdir -p $cell

    cd $cell; cp ../../submit.py .
    python submit.py --cell $cell --method $method \
                     --time 12:00:00 --cpus-per-task 32

    cd ..
done