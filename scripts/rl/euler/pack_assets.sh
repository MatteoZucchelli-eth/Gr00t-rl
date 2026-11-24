#!/bin/bash
#SBATCH --job-name=pack-assets
#SBATCH --output=logs/rl-pack-assets-%j.out
#SBATCH --error=logs/rl-pack-assets-%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
# =============================================================================
# pack_assets.sh — pack the MolmoSpaces asset tree into ONE tar file for fast job startup.
#
#   sbatch scripts/rl/euler/pack_assets.sh        (from ~/groot-rl; no GPU, no container)
#
# The asset tree is ~130k small files on Lustre (/cluster/work), and loading a scene reads
# thousands of them, which is slow on a shared parallel filesystem. submit_rl_euler.sh
# extracts this tarball to node-local $TMPDIR instead (one sequential read), whenever it
# exists. Re-run after the assets preset has downloaded scenes for new tasks.
# Writes to a temporary name and renames at the end, so jobs never see a partial tar.
# Env: MLSPACES_DIR (default /cluster/work/mrl/$USER/mlspaces),
#      MLSPACES_TAR (default /cluster/work/mrl/$USER/containers/mlspaces.tar).
# =============================================================================
set -euo pipefail

MLSPACES_DIR="${MLSPACES_DIR:-/cluster/work/mrl/$USER/mlspaces}"
MLSPACES_TAR="${MLSPACES_TAR:-/cluster/work/mrl/$USER/containers/mlspaces.tar}"
TMP_TAR="$MLSPACES_TAR.partial.${SLURM_JOB_ID:-$$}"

[ -d "$MLSPACES_DIR/assets" ] || { echo "no asset tree at $MLSPACES_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$MLSPACES_TAR")"
echo "[pack] $MLSPACES_DIR -> $MLSPACES_TAR ($(date))"
# Uncompressed: most bytes are already-compressed meshes/textures, and extraction must be
# fast. Lock files and in-flight downloads are left out; symlinks (which point into
# /mlspaces/cache, the path jobs bind the tree at) are kept as symlinks.
time tar -C "$MLSPACES_DIR" -b 2048 \
    --exclude='.lock' --exclude='*.lock' --exclude='*.incomplete' --exclude='*.tmp' \
    -cf "$TMP_TAR" .
mv -f "$TMP_TAR" "$MLSPACES_TAR"
ls -lh "$MLSPACES_TAR"
echo "[pack] done ($(date)); jobs now extract it to \$TMPDIR at start"
