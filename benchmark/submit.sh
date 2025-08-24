method="ref-kpts"
mkdir -p $method; cd $method

for cell in diamond; do
    mkdir -p $cell

    cd $cell; cp ../../submit.py .
    python submit.py --time="20:00:00" --cpus-per-task="32" \
    --reservation="changroup-h100-node-1"

    cd ..
done
