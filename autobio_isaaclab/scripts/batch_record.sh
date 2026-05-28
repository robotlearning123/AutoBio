#!/bin/bash
# Batch record all 12 AutoBio Isaac Lab tasks.
# Each task runs in its own process (required by Isaac Sim's singleton sim context).
set -e

TASKS=(
    "Isaac-AutoBio-Pickup-Direct-v0:pickup"
    "Isaac-AutoBio-Insert-Direct-v0:insert"
    "Isaac-AutoBio-ThermalCyclerOpen-Direct-v0:thermal_cycler_open"
    "Isaac-AutoBio-ThermalCyclerClose-Direct-v0:thermal_cycler_close"
    "Isaac-AutoBio-Pipette-Direct-v0:pipette"
    "Isaac-AutoBio-ScrewLoose-Direct-v0:screw_loose"
    "Isaac-AutoBio-ScrewTighten-Direct-v0:screw_tighten"
    "Isaac-AutoBio-Centrifuge5430-Direct-v0:centrifuge_5430"
    "Isaac-AutoBio-Centrifuge5910-Direct-v0:centrifuge_5910"
    "Isaac-AutoBio-CentrifugeMini-Direct-v0:centrifuge_mini"
    "Isaac-AutoBio-ThermalMixer-Direct-v0:thermal_mixer"
    "Isaac-AutoBio-VortexMixer-Direct-v0:vortex_mixer"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../demos"
mkdir -p "$OUT_DIR"

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

PASSED=0
FAILED=0
TOTAL=${#TASKS[@]}

for entry in "${TASKS[@]}"; do
    TASK_ID="${entry%%:*}"
    TASK_NAME="${entry##*:}"

    echo ""
    echo "============================================================"
    echo "Recording: $TASK_NAME ($TASK_ID)"
    echo "============================================================"

    conda run -n isaac5 --cwd "$SCRIPT_DIR/.." python scripts/record_single.py \
        --task "$TASK_ID" --name "$TASK_NAME" --headless --num_steps 30 2>&1 | \
        grep -E "Recording|reset OK|step|Wrote|WARNING|FAIL|No frames" || true

    # Check result
    RESULT_FILE="$OUT_DIR/${TASK_NAME}_result.json"
    if [ -f "$RESULT_FILE" ]; then
        OK=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('ok', False))")
        if [ "$OK" = "True" ]; then
            PASSED=$((PASSED + 1))
            echo "  => PASS"
        else
            FAILED=$((FAILED + 1))
            echo "  => FAIL"
        fi
    else
        FAILED=$((FAILED + 1))
        echo "  => FAIL (no result file)"
    fi
done

echo ""
echo "============================================================"
echo "SUMMARY: $PASSED/$TOTAL passed, $FAILED failed"
echo "============================================================"

# Check all mp4 outputs
echo ""
echo "Video files:"
for entry in "${TASKS[@]}"; do
    TASK_NAME="${entry##*:}"
    MP4="$OUT_DIR/${TASK_NAME}.mp4"
    if [ -f "$MP4" ]; then
        SIZE=$(du -h "$MP4" | cut -f1)
        echo "  $TASK_NAME.mp4 ($SIZE)"
    else
        echo "  $TASK_NAME.mp4 (MISSING)"
    fi
done
