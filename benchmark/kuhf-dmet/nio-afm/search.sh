for kmesh in 2-2-2 3-3-3 4-4-4 5-5-5 6-6-6; do
    log_file="./${kmesh}/fftisdf-180-25/"
    echo "log_file = $log_file"
    python search.py $log_file

    log_file="../nio-fm//${kmesh}/fftisdf-180-25/"
    echo "log_file = $log_file"
    python search.py $log_file
done