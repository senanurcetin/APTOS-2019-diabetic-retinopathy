#!/usr/bin/env bash
#
# Run the cross-validation sweep for one or more variants, one at a time.
#
#   bash scripts/run_cv.sh                                  # baseline squash clahe
#   bash scripts/run_cv.sh squash                           # just one
#   bash scripts/run_cv.sh --stratify diagnosis+resolution  # the confound-aware split
#
# The sweep is resumable: completed folds are skipped on a restart, so running
# this again after an interruption picks up where it stopped.
#
# Two rules, both learned the hard way - three earlier sweeps died to them:
#
#   1. Do not start a second GPU job while this runs. It is not VRAM that
#      breaks; peak usage is 2.2 GB of 6 GB at these settings. It is the
#      Windows commit limit, which dataloader worker processes push over.
#   2. Do not edit anything under src/aptos/ while this runs. Windows workers
#      re-import the main module by path, so a file changing underneath them
#      kills the run.
#
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore

STRATIFY="diagnosis"
if [[ "${1:-}" == "--stratify" ]]; then
    STRATIFY="$2"
    shift 2
fi

# Order matters if the run is interrupted: baseline is the reference every
# other comparison is made against, squash closes the gap the repository
# documents as unmeasured, and clahe re-tests a null result that already has
# three seeds behind it.
if (( $# )); then
    VARIANTS=("$@")
else
    VARIANTS=(baseline squash clahe)
fi

started_all=$(date +%s)
FAILED=()

for variant in "${VARIANTS[@]}"; do
    echo "============================================================"
    echo "variant: ${variant}   stratify: ${STRATIFY}   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    started=$(date +%s)

    # Deliberately not `set -e` around this. Aborting the whole sweep on one bad
    # variant is the opposite of what you want overnight - the others are still
    # worth having. Failures are collected and reported at the end.
    #
    # The old version of this script printed "(exit $?)" after an echo, so it
    # reported the status of the echo rather than of python, and every failure
    # looked like a success.
    if python -u -m aptos.training.cv \
        --config configs/cv.yaml \
        --variant "${variant}" \
        --stratify "${STRATIFY}"; then
        status="ok"
    else
        code=$?
        status="FAILED (exit ${code})"
        FAILED+=("${variant}")
    fi

    echo "--- ${variant}: ${status} after $(( ($(date +%s) - started) / 60 )) min"
    echo
done

echo "============================================================"
echo "sweep finished in $(( ($(date +%s) - started_all) / 60 )) min"
if (( ${#FAILED[@]} )); then
    echo "failed variants: ${FAILED[*]}"
    echo "re-run this script to resume - completed folds are skipped"
    exit 1
fi
echo "all variants completed"
