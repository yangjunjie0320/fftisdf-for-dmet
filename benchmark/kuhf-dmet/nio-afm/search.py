#!/usr/bin/env python3
"""
Extract DMET energies from slurm output files and update out.log files.

This script:
1. Finds all slurm-*.out files in the input folder
2. Extracts cycle 0 and last cycle energies from DMET iteration history
3. Updates the corresponding out.log file to verify ene_dmet matches last cycle
4. Adds/updates ene_dmet_0 with cycle 0 energy
"""

import argparse
import re
import sys
from pathlib import Path


def find_dmet_iteration_history(content):
    """
    Find the final DMET iteration history section in the file content.
    Returns the section content as a string, or None if not found.
    """
    # Find all DMET iteration history sections
    # Pattern matches: "|---------------------------- DMET iteration history ---------------------------|"
    # We search for this line and then include the header line before it
    pattern = r'\|-+\s+DMET iteration history\s+-+\|'
    matches = list(re.finditer(pattern, content))
    
    if not matches:
        return None
    
    # Get the last (final) section - this should be the converged one
    last_match = matches[-1]
    match_start = last_match.start()
    
    # Find the start of the section (the "@---------------@" line before the match)
    # Look backwards for the line starting with "@" and containing only "-" and "@"
    lines_before = content[:match_start].split('\n')
    section_start_pos = match_start
    for i in range(len(lines_before) - 1, max(0, len(lines_before) - 10), -1):
        line = lines_before[i]
        if line.startswith('@') and all(c in '@-' for c in line.rstrip()):
            # Found the header line, calculate its position
            section_start_pos = len('\n'.join(lines_before[:i]))
            break
    
    # Find the end of this section (the separator line before CPU time)
    # Look for the separator line: "---------------------------------------------------------------------------------"
    # This appears after the last cycle line
    remaining_content = content[section_start_pos:]
    separator_pattern = r'^-{70,}\n'
    separator_match = re.search(separator_pattern, remaining_content, re.MULTILINE)
    
    if separator_match:
        end_pos = section_start_pos + separator_match.end()
        return content[section_start_pos:end_pos]
    
    return None


def parse_energies_from_history(history_content):
    """
    Parse cycle 0 and last cycle energies from DMET iteration history.
    Returns (cycle_0_energy, last_cycle_energy) or (None, None) if not found.
    """
    # Pattern to match cycle lines: "    0      -367.3973088845       ..."
    # or "   27      -367.3923620517       ..."
    cycle_pattern = r'^\s+(\d+)\s+(-?\d+\.\d+)\s+'
    
    cycles = []
    for line in history_content.split('\n'):
        match = re.match(cycle_pattern, line)
        if match:
            cycle_num = int(match.group(1))
            energy = float(match.group(2))
            cycles.append((cycle_num, energy))
    
    if not cycles:
        return None, None
    
    # Get cycle 0 (first cycle)
    cycle_0_energy = None
    for cycle_num, energy in cycles:
        if cycle_num == 0:
            cycle_0_energy = energy
            break
    
    # Get last cycle (highest cycle number)
    last_cycle_num = max(cycle_num for cycle_num, _ in cycles)
    last_cycle_energy = None
    for cycle_num, energy in cycles:
        if cycle_num == last_cycle_num:
            last_cycle_energy = energy
            break
    
    return cycle_0_energy, last_cycle_energy


def read_out_log(out_log_path):
    """Read and parse out.log file. Returns a dict of key-value pairs."""
    data = {}
    with open(out_log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            # Parse "key = value" format
            parts = line.split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                data[key] = value
    return data


def write_out_log(out_log_path, data):
    """Write data dict to out.log file in key = value format."""
    with open(out_log_path, 'w') as f:
        for key, value in data.items():
            f.write(f"{key} = {value}\n")


def process_slurm_file(slurm_path):
    """Process a single slurm file and update corresponding out.log."""
    print(f"Processing {slurm_path}...")
    
    # Read slurm file
    with open(slurm_path, 'r') as f:
        content = f.read()
    
    # Extract DMET iteration history
    history_content = find_dmet_iteration_history(content)
    if history_content is None:
        print(f"  Warning: Could not find DMET iteration history in {slurm_path}")
        return False
    
    # Parse energies
    cycle_0_energy, last_cycle_energy = parse_energies_from_history(history_content)
    if cycle_0_energy is None or last_cycle_energy is None:
        print(f"  Warning: Could not parse energies from {slurm_path}")
        return False
    
    print(f"  Found cycle 0 energy: {cycle_0_energy}")
    print(f"  Found last cycle energy: {last_cycle_energy}")
    
    # Find out.log in the same directory
    out_log_path = slurm_path.parent / 'out.log'
    if not out_log_path.exists():
        print(f"  Warning: out.log not found at {out_log_path}")
        return False
    
    # Read out.log
    data = read_out_log(out_log_path)
    
    # Verify ene_dmet matches last cycle energy
    if 'ene_dmet' not in data:
        print(f"  Error: ene_dmet not found in {out_log_path}")
        return False
    
    try:
        ene_dmet = float(data['ene_dmet'])
    except ValueError:
        print(f"  Error: Could not parse ene_dmet value: {data['ene_dmet']}")
        return False
    
    # Check if energies match (allow small floating point error)
    if abs(ene_dmet - last_cycle_energy) > 1e-6:
        print(f"  Error: ene_dmet ({ene_dmet}) does not match last cycle energy ({last_cycle_energy})")
        print(f"  Difference: {abs(ene_dmet - last_cycle_energy)}")
        return False
    
    print(f"  Verified: ene_dmet matches last cycle energy")
    
    # Add/update ene_dmet_0
    data['ene_dmet_0'] = f"{cycle_0_energy:.11f}"
    print(f"  Added/updated ene_dmet_0 = {cycle_0_energy:.11f}")
    
    # Write updated out.log
    write_out_log(out_log_path, data)
    print(f"  Updated {out_log_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract DMET energies from slurm files and update out.log files'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Input folder containing slurm-*.out files'
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: Input path is not a directory: {input_folder}")
        sys.exit(1)
    
    # Find all slurm-*.out files in the input folder
    slurm_files = list(input_folder.glob('slurm-*.out'))
    
    if not slurm_files:
        print(f"Warning: No slurm-*.out files found in {input_folder}")
        return
    
    print(f"Found {len(slurm_files)} slurm file(s)")
    
    success_count = 0
    for slurm_file in sorted(slurm_files):
        if process_slurm_file(slurm_file):
            success_count += 1
        print()
    
    print(f"Successfully processed {success_count}/{len(slurm_files)} file(s)")


if __name__ == '__main__':
    main()
