#!/usr/bin/env bash
# run_all_tasks.sh -- run every known task through main.py (--model gpt), log each,
# print a task-layer/physical-layer summary. Skip tasks via SKIP="a b c".
cd "$(dirname "$0")"
mkdir -p outputs/_batch_logs

TASKS=$(python3 -c "
import glob, os
print(' '.join(sorted(os.path.splitext(os.path.basename(y))[0]
      for y in glob.glob('contrauto/tasks/*.yaml'))))
")

for t in $TASKS; do
    case " $SKIP " in *" $t "*) echo "=== $t (skipped) ==="; continue;; esac
    echo "=== $t ==="
    log="outputs/_batch_logs/${t}.log"
    python3 -u main.py --task "$t" --model gpt > "$log" 2>&1
    echo "exit=$? -> $log"
done

echo
echo "=== SUMMARY ==="
printf "%-16s %-10s %-10s\n" "task" "task-layer" "physical"
for t in $TASKS; do
    log="outputs/_batch_logs/${t}.log"
    [ -f "$log" ] || continue
    tl="?"; ph="-"
    grep -q "FSM produced\|FSM reuse" "$log" && tl="OK"
    grep -q "no FSM produced" "$log" && tl="FAIL"
    grep -q "no simulator registered" "$log" && ph="(none)"
    grep -q "✓  PHYSICAL EXECUTION" "$log" && ph="OK"
    grep -q "✗  PHYSICAL EXECUTION" "$log" && ph="FAIL"
    printf "%-16s %-10s %-10s\n" "$t" "$tl" "$ph"
done
