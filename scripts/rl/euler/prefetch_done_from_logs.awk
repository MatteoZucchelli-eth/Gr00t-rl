# Tasks finished by a prefetch run, from its log (scripts/rl/prefetch_molmospaces.py output).
# A task is done if its "start" line is followed by another "start" line and no failure
# was logged for it; the last task of a killed job is therefore redone.
#   awk -f prefetch_done_from_logs.awk logs/rl-assets-*.out | sort -u
/\[prefetch\] start / {
    if (prev != "" && !(prev in bad)) print prev
    prev = $3
}
/\[prefetch\]   [^ ]+: / && $3 != "ok" { t = $2; sub(":", "", t); bad[t] = 1 }
/\[prefetch\] FAILED / { t = $3; sub(":", "", t); bad[t] = 1 }
