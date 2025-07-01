cell="diamond"
mkdir -p $cell

cd $cell; cp ../submit.py .
python submit.py --cell $cell --method kpt --ntasks 1 --time 20:00:00

cd ..

