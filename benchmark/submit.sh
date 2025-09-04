method="krhf-dmet"
mkdir -p $method; cd $method

for cell in co2; do
    mkdir -p $cell

    cd $cell; cp /home/junjiey/work/fftisdf-for-dmet/src/code/submit.py .
    python submit.py --time="20:00:00" --cpus-per-task="32" \
    --reservation="changroup_standingres"

    cd ..
done
