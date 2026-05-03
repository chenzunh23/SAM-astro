#!/usr/bin/env bash
set -euo pipefail

HOST_PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-lsst_distrib_eval_recreated}"
LOAD_LSST="${LOAD_LSST:-/opt/lsst/software/stack/loadLSST.bash}"
LSST_TOP_PACKAGE="${LSST_TOP_PACKAGE:-lsst_distrib}"
DOCKER_CMD=(${DOCKER_CMD:-docker})

if ! "${DOCKER_CMD[@]}" info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    DOCKER_CMD=(sudo docker)
  fi
fi

if ! "${DOCKER_CMD[@]}" info >/dev/null 2>&1; then
  cat <<'EOF' >&2
Unable to reach the Docker daemon from this account.

If you see a permission denied error on /var/run/docker.sock, either:
  - add this user to the docker group and log out/in, or
  - rerun with DOCKER_CMD='sudo docker' if sudo access is available.
EOF
  exit 1
fi

CONTAINER_PACKAGE_ROOT="${CONTAINER_PACKAGE_ROOT:-/tmp/scarlet_deblend_demo_package_$$}"
INPUT_ROOT="${INPUT_ROOT:-${CONTAINER_PACKAGE_ROOT}/input/projection_cutout}"
REPO="${REPO:-${CONTAINER_PACKAGE_ROOT}/input/repo}"
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-${HOST_PACKAGE_ROOT}/output/scarlet_demo_test}"
CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR:-/tmp/scarlet_deblend_demo_output_$$}"
TRACT="${TRACT:-9813}"
PATCH="${PATCH:-4,5}"

"${DOCKER_CMD[@]}" exec "${CONTAINER_NAME}" /bin/bash -lc "mkdir -p '${CONTAINER_PACKAGE_ROOT}'"
"${DOCKER_CMD[@]}" cp "${HOST_PACKAGE_ROOT}/input" "${CONTAINER_NAME}:${CONTAINER_PACKAGE_ROOT}/input"
"${DOCKER_CMD[@]}" cp "${HOST_PACKAGE_ROOT}/scarlet_deblend_from_fits.py" "${CONTAINER_NAME}:${CONTAINER_PACKAGE_ROOT}/scarlet_deblend_from_fits.py"

"${DOCKER_CMD[@]}" exec "${CONTAINER_NAME}" /bin/bash -lc "
  set -eo pipefail
  source '${LOAD_LSST}' >/dev/null 2>&1
  setup '${LSST_TOP_PACKAGE}' >/dev/null 2>&1
  python -u '${CONTAINER_PACKAGE_ROOT}/scarlet_deblend_from_fits.py' \
    --repo '${REPO}' \
    --tract '${TRACT}' \
    --patch '${PATCH}' \
    --coadd HSC-G='${INPUT_ROOT}/HSC-G/deepCoadd-HSC-G-${TRACT}-${PATCH}.fits' \
    --coadd HSC-R='${INPUT_ROOT}/HSC-R/deepCoadd-HSC-R-${TRACT}-${PATCH}.fits' \
    --coadd HSC-I='${INPUT_ROOT}/HSC-I/deepCoadd-HSC-I-${TRACT}-${PATCH}.fits' \
    --clip-sky-sources-to-exposure-bbox \
    --output-dir '${CONTAINER_OUTPUT_DIR}'
"

rm -rf "${HOST_OUTPUT_DIR}"
mkdir -p "${HOST_OUTPUT_DIR}"
"${DOCKER_CMD[@]}" cp "${CONTAINER_NAME}:${CONTAINER_OUTPUT_DIR}/." "${HOST_OUTPUT_DIR}"
echo "Wrote demo output under: ${HOST_OUTPUT_DIR}"
