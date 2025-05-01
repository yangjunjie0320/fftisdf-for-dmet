import os
import re
from collections import defaultdict

def to_float(s: str):
    try:
        return float(s)
    except ValueError:
        return s

def collect_log_data(root_dir='.', file_name='out.log'):
    """
    Collect data from all out.log files recursively and return as a dictionary.
    
    Args:
        root_dir (str): Root directory to search from
    
    Returns:
        dict: Nested dictionary with the collected data
    """
    results = {}
    
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if not f == file_name:
                continue
            filepath = os.path.join(dirpath, f)
            
            data = {}
            lines = open(filepath, 'r').readlines()
            for line in lines:
                if '=' in line:
                    key, value = line.split('=')
                    key = key.strip()
                    value = value.strip()
                    data[key] = to_float(value)

                elif 'Time for' in line:
                    left, right = line.split(':')
                    key = "time_" + left.split()[2]
                    value = right.split()[0]
                    data[key] = to_float(value)

            results[dirpath] = data

    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--filename', type=str, default='out.log')
    parser.add_argument('--output', type=str, default='collect.json')
    args = parser.parse_args()

    import json
    results = collect_log_data(args.root_dir, args.filename)
    with open(args.output, 'w') as f:
        json.dump(results, f)
    
    # python collect.py --root_dir=../benchmark/nio-afm/ --filename=out.log --output=./data/nio-afm-dzvp.json
    # python collect.py --root_dir=../benchmark/diamond/ --filename=out.log --output=./data/diamond-dzvp.json