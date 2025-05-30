cell="diamond"
method="klno"

scancel -u $USER
if [ -d $cell/$method ]; then
    rm -rf $cell/$method
fi
mkdir -p $cell/$method
cd $cell/$method

cp ../../submit.py .
python submit.py --cell $cell --method $method --ntasks 1 --time 04:00:00
