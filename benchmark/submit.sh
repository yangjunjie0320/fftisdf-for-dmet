method="gamma"
mkdir -p $method; cd $method

for cell in diamond co2; do
    mkdir -p $cell

    cd $cell; cp ../../submit.py .
    python submit.py --cell $cell --method $method \
                     --time 00:30:00 --cpus-per-task 4

    cd ..
done