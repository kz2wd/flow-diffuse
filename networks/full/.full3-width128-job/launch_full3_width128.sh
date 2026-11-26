#!/bin/sh
set -eu
job_dir=networks/full/.full3-width128-job
python_path=$1
mkdir "$job_dir/started.lock"
printf '%s\n' "$$" > "$job_dir/pid"
code=0
MPLBACKEND=Agg "$python_path" -u "$job_dir/run_full3_width128.py" --repo . --output "$job_dir/results" || code=$?
printf '%s\n' "$code" > "$job_dir/exit.tmp"
mv "$job_dir/exit.tmp" "$job_dir/exit_code"
exit "$code"
