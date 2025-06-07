# scancel -u $USER

cell="diamond"
mkdir -p $cell

cd $cell; cp ../submit.py .

for method in "kmp2" "klno"; do
    python submit.py --cell $cell --method $method --ntasks 1 --time 20:00:00
done

cd ..

