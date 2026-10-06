#!/usr/bin/env bash
# Run on login node: read-only account/resource inventory, no fitting.
set -u
whoami
pwd
command -v myquota >/dev/null && myquota
command -v sinfo >/dev/null && sinfo -o '%P %a %l %D %c %m'
command -v sacctmgr >/dev/null && sacctmgr -n show associations user="$USER" format=Account,Partition,QOS
if type module >/dev/null 2>&1; then
  module -t avail Python 2>&1
  module -t avail FAISS 2>&1
fi
command -v python3
python3 --version
printf '%s\n' 'History only; check directory exists and is still yours:'
ls -ld /project/home/p201605/u104824/actionpiece_exp 2>/dev/null || true
printf '%s\n' 'Return account, partition, permitted QOS, Python/modules, repo path and limits; do not infer them from history.'
