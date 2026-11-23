#!/bin/bash
# Shared SSH helpers for the Euler scripts (sourced, not executed).
#
# Euler's login load balancer (eu-lb.euler.ethz.ch) occasionally closes a new connection
# before authentication ("Connection closed by 129.132.93.72 port 22"). So we:
#   * open ONE multiplexed master connection and reuse it for every ssh/rsync call;
#   * retry on connection failures (ssh/rsync exit code 255; rsync 10/12/30/35 = I/O or
#     timeout errors), with --partial transfers resuming where they stopped.
# Env: EULER_HOST (default euler), EULER_RETRIES (default 6).

EULER_HOST="${EULER_HOST:-euler}"
EULER_RETRIES="${EULER_RETRIES:-6}"
_EULER_CTL="${XDG_RUNTIME_DIR:-/tmp}/euler-ssh-%r@%h:%p"
EULER_SSH_OPTS=(
    -o ControlMaster=auto -o "ControlPath=$_EULER_CTL" -o ControlPersist=10m
    -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6
)

_euler_retry() {
    # _euler_retry <cmd...>: rerun on connection-type failures, with backoff.
    local attempt=1 rc
    while true; do
        "$@" && return 0
        rc=$?
        case "$rc" in
            255 | 10 | 12 | 30 | 35) ;;  # connection / protocol / timeout errors: retry
            *) return "$rc" ;;
        esac
        if [ "$attempt" -ge "$EULER_RETRIES" ]; then
            echo "[euler] giving up after $attempt attempts (exit $rc)" >&2
            return "$rc"
        fi
        echo "[euler] connection problem (exit $rc), retry $attempt/$((EULER_RETRIES - 1)) in $((attempt * 5))s..." >&2
        ssh -O exit "${EULER_SSH_OPTS[@]}" "$EULER_HOST" >/dev/null 2>&1 || true  # drop a stale master
        sleep $((attempt * 5))
        attempt=$((attempt + 1))
    done
}

euler_connect() {
    # Establish the master connection up front (retried).
    _euler_retry ssh "${EULER_SSH_OPTS[@]}" -o BatchMode=yes "$EULER_HOST" true
}

euler_ssh() {
    _euler_retry ssh "${EULER_SSH_OPTS[@]}" "$EULER_HOST" "$@"
}

euler_rsync() {
    # euler_rsync <rsync args...>; remote paths written as "$EULER_HOST:..."
    _euler_retry rsync --partial -e "ssh ${EULER_SSH_OPTS[*]}" "$@"
}
