#!/bin/bash
# Run one experiment: ./run_one.sh "description"
DESC="$1"
PYTHON=".venv/bin/python"

git add train.py prepare.py 2>/dev/null
git commit -m "exp: $DESC" --allow-empty 2>/dev/null
COMMIT=$(git rev-parse --short HEAD)

OUTPUT=$($PYTHON train.py 2>&1)
EC=$?

if [ $EC -ne 0 ]; then
    echo "${COMMIT}	0.000000	0.000000	0.000000	16.3	crash	${DESC}" >> results.tsv
    echo "CRASHED (exit=$EC)"
    echo "$OUTPUT" | tail -5
    exit 1
fi

VL=$(echo "$OUTPUT" | grep "^val_loss:" | awk '{print $2}')
TB=$(echo "$OUTPUT" | grep "^text_bpb:" | awk '{print $2}')
AL=$(echo "$OUTPUT" | grep "^audio_loss:" | awk '{print $2}')

if [ -z "$VL" ] || [ "$AL" = "0.000000" ]; then
    echo "${COMMIT}	${VL:-0}	${TB:-0}	${AL:-0}	16.3	crash	${DESC}" >> results.tsv
    echo "FAILED: VL=$VL TB=$TB AL=$AL"
    echo "$OUTPUT" | tail -10
    exit 1
fi

echo "${COMMIT}	${VL}	${TB}	${AL}	16.3	keep	${DESC}" >> results.tsv
echo "OK: val_loss=${VL} text_bpb=${TB} audio_loss=${AL}"
