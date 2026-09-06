#!/usr/bin/env bash
# Export ALL public + private GPG keyring material for user "sweeden"
# and store it permanently under a root-owned directory.
#
# Intended invocation (on a host where user sweeden exists):
#   sudo /path/to/export_sweeden_gpg_keyrings_as_root.sh
#
# Permanent store (default):
#   /root/permanent/gpg/sweeden/<UTC-timestamp>/
#
# Layout per GNUPGHOME discovered:
#   <label>/
#     public_keys.asc          # all public keys (armored)
#     private_keys.asc         # all secret keys (armored) — mode 0600
#     ownertrust.txt           # trust database export
#     keyring_raw/             # verbatim copy of keyring files
#     inventory.txt            # fingerprints / UIDs (no secret material)
#     checksums.sha256
#
# WARNING: private_keys.asc is full secret-key material. Keep root-only.

set -euo pipefail

SOURCE_USER="${SOURCE_USER:-sweeden}"
DEST_ROOT="${DEST_ROOT:-/root/permanent/gpg/${SOURCE_USER}}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${DEST_ROOT}/${STAMP}"
LOG="${DEST}/export.log"

die() { echo "ERROR: $*" >&2; exit 1; }

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    die "run as root (e.g. sudo $0). Current uid=$(id -u)."
  fi
}

require_tools() {
  command -v gpg >/dev/null 2>&1 || die "gpg not found"
  command -v getent >/dev/null 2>&1 || die "getent not found"
  command -v rsync >/dev/null 2>&1 || command -v cp >/dev/null 2>&1 || die "rsync/cp required"
}

source_home() {
  local home
  home="$(getent passwd "${SOURCE_USER}" | awk -F: '{print $6}')"
  [[ -n "${home}" ]] || die "user '${SOURCE_USER}' not found in passwd"
  [[ -d "${home}" ]] || die "home directory missing: ${home}"
  printf '%s\n' "${home}"
}

# Discover GNUPGHOME directories belonging to sweeden (primary + competition rings).
discover_gnupg_homes() {
  local home="$1"
  local -a candidates=()
  candidates+=("${home}/.gnupg")
  # Competition / challenge keyrings commonly used in this project
  candidates+=(
    "${home}/kagg/reasoning_vs_questioning/artifacts/elon_musk/gnupg"
    "${home}/kagg/reasoning_vs_questioning/artifacts/scott_weeden/keyring/gnupg"
    "${home}/kagg/reasoning_vs_questioning/artifacts/cursor/keyring"
  )
  # Allow operator override: colon-separated extra homes
  if [[ -n "${EXTRA_GNUPGHOMES:-}" ]]; then
    IFS=':' read -r -a extras <<<"${EXTRA_GNUPGHOMES}"
    candidates+=("${extras[@]}")
  fi

  local c
  for c in "${candidates[@]}"; do
    # Only treat directories that look like a GnuPG home as keyrings.
    if [[ -d "${c}" ]] && {
      [[ -e "${c}/pubring.kbx" ]] \
        || [[ -e "${c}/pubring.gpg" ]] \
        || [[ -d "${c}/private-keys-v1.d" ]] \
        || [[ -e "${c}/trustdb.gpg" ]]
    }; then
      printf '%s\n' "${c}"
    fi
  done
}

safe_label() {
  # Turn an absolute path into a filesystem-safe label
  local p="$1"
  printf '%s' "${p}" | sed 's#^/##' | tr '/ ' '__'
}

export_one_home() {
  local gnupghome="$1"
  local label
  label="$(safe_label "${gnupghome}")"
  local out="${DEST}/${label}"
  mkdir -p "${out}/keyring_raw"
  chmod 700 "${out}" "${out}/keyring_raw"

  echo "=== Exporting GNUPGHOME=${gnupghome} -> ${out} ===" | tee -a "${LOG}"

  # Inventory (public metadata only)
  {
    echo "# inventory for ${gnupghome}"
    echo "# generated ${STAMP}"
    echo
    echo "## public keys"
    GNUPGHOME="${gnupghome}" gpg --list-keys --with-fingerprint --with-keygrip 2>/dev/null || true
    echo
    echo "## secret keys"
    GNUPGHOME="${gnupghome}" gpg --list-secret-keys --with-fingerprint --with-keygrip 2>/dev/null || true
  } >"${out}/inventory.txt"
  chmod 600 "${out}/inventory.txt"

  # Armored public keys (all)
  if GNUPGHOME="${gnupghome}" gpg --batch --export --armor >"${out}/public_keys.asc" 2>>"${LOG}"; then
    chmod 644 "${out}/public_keys.asc"
  else
    echo "(no public keys or export failed)" | tee -a "${LOG}"
    : >"${out}/public_keys.asc"
    chmod 644 "${out}/public_keys.asc"
  fi

  # Armored private/secret keys (all) — highly sensitive
  if GNUPGHOME="${gnupghome}" gpg --batch --export-secret-keys --armor >"${out}/private_keys.asc" 2>>"${LOG}"; then
    chmod 600 "${out}/private_keys.asc"
  else
    echo "(no secret keys or export failed)" | tee -a "${LOG}"
    : >"${out}/private_keys.asc"
    chmod 600 "${out}/private_keys.asc"
  fi

  # Also export secret subkeys explicitly (some setups need this)
  GNUPGHOME="${gnupghome}" gpg --batch --export-secret-subkeys --armor \
    >"${out}/private_subkeys.asc" 2>>"${LOG}" || : >"${out}/private_subkeys.asc"
  chmod 600 "${out}/private_subkeys.asc"

  # Ownertrust
  GNUPGHOME="${gnupghome}" gpg --batch --export-ownertrust \
    >"${out}/ownertrust.txt" 2>>"${LOG}" || : >"${out}/ownertrust.txt"
  chmod 600 "${out}/ownertrust.txt"

  # Raw keyring files (complete machine-readable backup)
  # Includes pubring.kbx / pubring.gpg, trustdb, private-keys-v1.d, openpgp-revocs.d
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      --exclude='S.gpg-agent*' \
      --exclude='S.scdaemon*' \
      --exclude='*.lock' \
      "${gnupghome}/" "${out}/keyring_raw/"
  else
    cp -a "${gnupghome}/." "${out}/keyring_raw/" 2>/dev/null || true
    rm -f "${out}/keyring_raw"/S.gpg-agent* "${out}/keyring_raw"/S.scdaemon* 2>/dev/null || true
  fi
  chmod -R go-rwx "${out}/keyring_raw"

  # Checksums
  (
    cd "${out}"
    find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum
  ) >"${out}/checksums.sha256"
  chmod 600 "${out}/checksums.sha256"

  echo "OK ${label}" | tee -a "${LOG}"
}

main() {
  require_root
  require_tools

  local home
  home="$(source_home)"

  mkdir -p "${DEST}"
  chmod 700 "${DEST_ROOT}" 2>/dev/null || true
  chmod 700 "${DEST}"
  touch "${LOG}"
  chmod 600 "${LOG}"

  {
    echo "export_started=${STAMP}"
    echo "source_user=${SOURCE_USER}"
    echo "source_home=${home}"
    echo "dest=${DEST}"
    echo "gpg=$(gpg --version | head -1)"
    echo "host=$(hostname 2>/dev/null || echo unknown)"
  } | tee -a "${LOG}"

  local homes=()
  mapfile -t homes < <(discover_gnupg_homes "${home}" | sort -u)
  if [[ "${#homes[@]}" -eq 0 ]]; then
    die "no GNUPGHOME directories found for ${SOURCE_USER} under ${home}"
  fi

  echo "discovered_homes=${#homes[@]}" | tee -a "${LOG}"
  local h
  for h in "${homes[@]}"; do
    export_one_home "${h}"
  done

  # Manifest of this export run
  {
    echo "stamp=${STAMP}"
    echo "source_user=${SOURCE_USER}"
    echo "dest=${DEST}"
    echo "homes:"
    for h in "${homes[@]}"; do
      echo "  - ${h}"
    done
  } >"${DEST}/MANIFEST.txt"
  chmod 600 "${DEST}/MANIFEST.txt"

  # Pointer to latest export for permanent convenience
  ln -sfn "${DEST}" "${DEST_ROOT}/latest"
  chmod 700 "${DEST_ROOT}"

  echo "COMPLETE permanent store: ${DEST}" | tee -a "${LOG}"
  echo "LATEST symlink: ${DEST_ROOT}/latest"
}

main "$@"
