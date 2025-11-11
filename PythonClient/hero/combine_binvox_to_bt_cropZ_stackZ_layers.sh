#!/bin/bash
set -euo pipefail

# ====== CONFIG ======
BINVOX_DIR="/home/sgarimella34/multi-robot-coordination/data_binvox_octomap/ausenv_semanticrag_1mcubed/"
BINVOX2BT="/home/sgarimella34/octomap/bin/binvox2bt_unique_offsets"
BT_DIR="$BINVOX_DIR"                           # where layer_*.bt will be written
PATCH_SIZE=100                                 # must match Python collector

# BBOX="xmin ymin zmin xmax ymax zmax"
# BBOX="-500 -500 -600 500 500 600"              # crop box for octomap
BBOX="-1000 -1000 -600 1000 1000 600"
# =====================

usage() {
  cat <<EOF
Usage: $0 [layer_indices_or_ranges...]

If no indices are supplied, all layers found in BINVOX_DIR are processed.
Examples:
  $0              # build all layers
  $0 0 2 5        # build only layers 0, 2, and 5
  $0 0-3 7-9      # build 0,1,2,3 and 7,8,9
EOF
}

# ---- collect list of layers to process ----
declare -a REQUESTED_LAYERS=()

if (( $# == 0 )); then
  # auto-detect all layer indices
  shopt -s nullglob
  for f in "$BINVOX_DIR"/patch_*_layer*.binvox; do
    base=$(basename "$f" .binvox)
    if [[ $base =~ _layer([0-9]+)$ ]]; then
      REQUESTED_LAYERS+=( "${BASH_REMATCH[1]}" )
    fi
  done
  shopt -u nullglob
else
  # parse args (single index or range a-b)
  for spec in "$@"; do
    if [[ $spec =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start=${BASH_REMATCH[1]}
      end=${BASH_REMATCH[2]}
      if (( end < start )); then
        echo "Range $spec is invalid (end < start)" >&2
        exit 1
      fi
      for ((i=start; i<=end; i++)); do
        REQUESTED_LAYERS+=( "$i" )
      done
    elif [[ $spec =~ ^[0-9]+$ ]]; then
      REQUESTED_LAYERS+=( "$spec" )
    else
      echo "Invalid layer spec: $spec" >&2
      usage
      exit 1
    fi
  done
fi

# unique + sort numeric
IFS=$'\n' REQUESTED_LAYERS=($(printf "%s\n" "${REQUESTED_LAYERS[@]}" | sort -n | uniq))
unset IFS

if (( ${#REQUESTED_LAYERS[@]} == 0 )); then
  echo "No layers to process." >&2
  exit 1
fi

echo "Building ${#REQUESTED_LAYERS[@]} layer bt files..."
printf '  layer_%s\n' "${REQUESTED_LAYERS[@]}"

# ---- build each layer separately ----
for iz in "${REQUESTED_LAYERS[@]}"; do
  out_bt="$BT_DIR/layer_${iz}.bt"

  echo "== Layer $iz -> $out_bt =="

  # Start command for this layer
  CMD=( "$BINVOX2BT" --mark-free --bb $BBOX )

  shopt -s nullglob
  matched=false
  for f in "$BINVOX_DIR"/patch_*_layer${iz}.binvox; do
    matched=true
    base=$(basename "$f" .binvox)
    # pattern: patch_<cx>_<cy>_layer<iz>.binvox
    if [[ $base =~ ^patch_([^_]*)_([^_]*)_layer([0-9]+)$ ]]; then
      cx="${BASH_REMATCH[1]}"
      cy="${BASH_REMATCH[2]}"
      # iz from filename already known

      # SAME XY transform you used before
      ox=$(echo "$cy" | bc -l)
      oy=$(echo "$cx" | bc -l)

      # Collector saved centers at 0, -100, -200 ... (NED up = negative),
      # To put them "up" in OctoMap (positive), flip sign here:
      oz=$(echo "$iz * $PATCH_SIZE" | bc -l)

      CMD+=( --offset "$ox" "$oy" "$oz" "$f" )
    else
      echo "Skipping unexpected filename: $base" >&2
    fi
  done
  shopt -u nullglob

  if ! $matched; then
    echo "Warning: No binvox files found for layer $iz, skipping." >&2
    continue
  fi

  CMD+=( -o "$out_bt" )

  echo "Running:"
  printf '%q ' "${CMD[@]}"; echo
  "${CMD[@]}"
done

echo "Done. Individual layer bt files are in $BT_DIR"
