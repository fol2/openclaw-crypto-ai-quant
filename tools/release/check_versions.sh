#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
version_file="${repo_root}/VERSION"

[[ -f "${version_file}" ]] || die "Missing VERSION file at ${version_file}"

version="$(tr -d '[:space:]' < "${version_file}")"
[[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "VERSION must match SemVer X.Y.Z (got '${version}')"

mapfile -t cargo_tomls < <(find "${repo_root}" -type f -name Cargo.toml \
  ! -path "${repo_root}/target/*" \
  ! -path "${repo_root}/.git/*" | sort)
(( ${#cargo_tomls[@]} > 0 )) || die "No Cargo.toml files found under ${repo_root}"

for cargo_toml in "${cargo_tomls[@]}"; do
  crate_version="$(awk -F'"' '/^version = "/ { print $2; exit }' "${cargo_toml}")"
  [[ -n "${crate_version}" ]] || continue
  [[ "${crate_version}" == "${version}" ]] || die "Crate version '${crate_version}' != VERSION '${version}' in ${cargo_toml}"
done

if [[ "${GITHUB_REF_TYPE:-}" == "tag" ]]; then
  tag_name="${GITHUB_REF_NAME:-}"
  [[ "${tag_name}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Tag '${tag_name}' must match vX.Y.Z"
  [[ "${tag_name#v}" == "${version}" ]] || die "Tag '${tag_name}' != VERSION '${version}'"
fi

if git -C "${repo_root}" describe --tags --exact-match >/dev/null 2>&1; then
  head_tag="$(git -C "${repo_root}" describe --tags --exact-match)"
  if [[ "${head_tag}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    [[ "${head_tag#v}" == "${version}" ]] || die "HEAD tag '${head_tag}' != VERSION '${version}'"
  fi
fi

echo "Version governance check passed for ${version}."
