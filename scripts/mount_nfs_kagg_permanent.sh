#!/usr/bin/env bash
# Permanent NFS mount of 192.168.0.15 → /home/tmp/kagg
#
# Intended for LAN hosts (e.g. Cursor My Machines worker cpu-4-1 / mlx-4-16).
# Will fail fast on Cursor-managed cloud VMs that cannot reach 192.168.0.15.
#
# Usage:
#   sudo bash scripts/mount_nfs_kagg_permanent.sh
#   sudo NFS_EXPORT=/path/on/server bash scripts/mount_nfs_kagg_permanent.sh

set -euo pipefail

NFS_HOST="${NFS_HOST:-192.168.0.15}"
MOUNT_POINT="${MOUNT_POINT:-/home/tmp/kagg}"
FSTYPE="${FSTYPE:-nfs}"
MOUNT_OPTS="${MOUNT_OPTS:-defaults,_netdev,nofail,x-systemd.automount}"
NFS_EXPORT="${NFS_EXPORT:-}"

log() { printf '[mount_nfs_kagg] %s\n' "$*"; }
die() { printf '[mount_nfs_kagg] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(id -u)" -eq 0 ]] || die "run as root (sudo)"

if ! ping -c 1 -W 2 "$NFS_HOST" >/dev/null 2>&1; then
  die "$NFS_HOST unreachable — run this on a LAN host (e.g. machine=cpu-4-1), not a managed cloud VM"
fi

if ! command -v mount.nfs >/dev/null 2>&1 && ! command -v mount.nfs4 >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq nfs-common
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y nfs-utils
  elif command -v yum >/dev/null 2>&1; then
    yum install -y nfs-utils
  else
    die "nfs client tools missing and no supported package manager found"
  fi
fi

pick_export() {
  local exports line path preferred
  if [[ -n "$NFS_EXPORT" ]]; then
    printf '%s\n' "$NFS_EXPORT"
    return 0
  fi
  if ! command -v showmount >/dev/null 2>&1; then
    die "showmount missing; set NFS_EXPORT=/remote/path explicitly"
  fi
  exports="$(showmount -e "$NFS_HOST" 2>/dev/null | awk 'NR>1 {print $1}')" \
    || die "showmount -e $NFS_HOST failed"
  [[ -n "$exports" ]] || die "no NFS exports advertised by $NFS_HOST"

  preferred=""
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    case "$path" in
      */kagg|*/kagg/*|/home/tmp/kagg|/export/kagg|/kagg)
        preferred="$path"
        break
        ;;
    esac
  done <<<"$exports"

  if [[ -z "$preferred" ]]; then
    # sole export, else first line
    preferred="$(printf '%s\n' "$exports" | awk 'NF{print; exit}')"
    if [[ "$(printf '%s\n' "$exports" | awk 'NF' | wc -l | tr -d ' ')" -gt 1 ]]; then
      log "multiple exports; defaulting to first: $preferred (override with NFS_EXPORT=...)"
      printf '%s\n' "$exports" | sed 's/^/  - /' >&2
    fi
  fi
  printf '%s\n' "$preferred"
}

EXPORT_PATH="$(pick_export)"
REMOTE="${NFS_HOST}:${EXPORT_PATH}"
FSTAB_LINE="${REMOTE}  ${MOUNT_POINT}  ${FSTYPE}  ${MOUNT_OPTS}  0  0"

mkdir -p "$MOUNT_POINT"

# Idempotent fstab update for this mount point.
if [[ -f /etc/fstab ]] && grep -Eq "[[:space:]]${MOUNT_POINT}[[:space:]]" /etc/fstab; then
  log "updating existing fstab entry for ${MOUNT_POINT}"
  tmp="$(mktemp)"
  awk -v mp="$MOUNT_POINT" -v line="$FSTAB_LINE" '
    $2 == mp { print line; next }
    { print }
  ' /etc/fstab >"$tmp"
  cp "$tmp" /etc/fstab
  rm -f "$tmp"
else
  log "appending fstab entry"
  printf '\n# permanent kagg NFS (scripts/mount_nfs_kagg_permanent.sh)\n%s\n' \
    "$FSTAB_LINE" >>/etc/fstab
fi

# If already mounted from a different source, leave it and report.
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
  current="$(findmnt -n -o SOURCE --target "$MOUNT_POINT" || true)"
  if [[ "$current" == "$REMOTE" ]]; then
    log "already mounted: $current → $MOUNT_POINT"
  else
    die "$MOUNT_POINT already mounted from '$current' (wanted '$REMOTE'); unmount manually first"
  fi
else
  mount "$MOUNT_POINT" || mount -t "$FSTYPE" -o "$MOUNT_OPTS" "$REMOTE" "$MOUNT_POINT"
fi

log "fstab: $FSTAB_LINE"
findmnt "$MOUNT_POINT" || true
df -h "$MOUNT_POINT"
ls -la "$MOUNT_POINT" | head
log "OK: permanent NFS mount ready at $MOUNT_POINT"
