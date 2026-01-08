for kmesh in 2-2-2 3-3-3 4-4-4 5-5-5 6-6-6; do
    echo "";
    log_file="../nio-afm/${kmesh}/fftisdf-180-25/"
    echo "log_file = $log_file"
    grep -A1 "charge of    0Ni1" $log_file/sl* | tail -2
    grep "cycle               energy" $log_file/sl* | head -1
    grep -B6 "DMET converged after" $log_file/sl* | head -1
    grep "ene_dmet" $log_file/out.log

    log_file="../nio-fm/${kmesh}/fftisdf-180-25/"
    echo "log_file = $log_file"
    grep -A1 "charge of    0Ni1" $log_file/sl* | tail -2
    grep "cycle               energy" $log_file/sl* | head -1
    grep -B6 "DMET converged after" $log_file/sl* | head -1
    grep "ene_dmet" $log_file/out.log
done
