method="gamma"
mkdir -p $method; cd $method

for cell in diamond silicon nh3 co2; do
    mkdir -p $cell

    cd $cell; cp ../../submit.py .
    python submit.py --cell $cell --method $method \
                     --ntasks 1 --time 00:30:00

    cd ..
done