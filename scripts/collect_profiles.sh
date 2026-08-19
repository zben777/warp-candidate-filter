#!/usr/bin/env bash

set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: ./scripts/collect_profiles.sh [--sudo-ncu]

Rebuild and collect one run plus Nsight Compute report files for every kernel
version. Use --sudo-ncu when GPU performance counters require administrator
privileges.
EOF
}

use_sudo_ncu=0
case "${1:-}" in
    "") ;;
    --sudo-ncu) use_sudo_ncu=1 ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
profiles_dir="${repo_dir}/profiles"
staging_dir="$(mktemp -d "${repo_dir}/.profiles-staging.XXXXXX")"
backup_dir="${repo_dir}/.profiles-backup.$$"

cleanup()
{
    if [[ -n "${staging_dir:-}" && -d "${staging_dir}" ]]; then
        rm -rf "${staging_dir}"
    fi
}
trap cleanup EXIT

versions=(
    v0
    v0_shared
    v1
    v1.5
    v2
    v3
    v4
    v5
    v6
    v7
    v8
)

declare -A kernels=(
    [v0]=brute_force_real_kernel
    [v0_shared]=brute_force_shared_kernel
    [v1]=binary_search_real_kernel
    [v1.5]=v2_register_gather_compact_kernel
    [v2]=v2_2way_ilp_kernel
    [v3]=v3_fixed32_kernel
    [v4]=v4_fixed32_2way_ilp_kernel
    [v5]=v5_block_coalesced_store_kernel
    [v6]=v6_subwarp_int4_kernel
    [v7]=v7_subwarp_cp_async_kernel
    [v8]=v8_subwarp_early_cp_async_kernel
)

ncu_path="$(command -v ncu || true)"
if [[ -z "${ncu_path}" ]]; then
    printf 'Nsight Compute CLI (ncu) was not found in PATH.\n' >&2
    exit 1
fi

if ((use_sudo_ncu)); then
    printf 'Validating sudo credentials for Nsight Compute...\n'
    sudo -v
    ncu_command=(sudo "${ncu_path}")
else
    ncu_command=("${ncu_path}")
fi

printf 'Building all sm_89 executables...\n'
make -C "${repo_dir}" all

for version in "${versions[@]}"; do
    source_executable="${repo_dir}/build/sm_89/${version}"
    version_dir="${staging_dir}/${version}"
    artifact_name="${version}_real_4b"
    archived_executable="${version_dir}/${artifact_name}"
    run_result="${version_dir}/${artifact_name}_result.txt"
    ncu_report_base="${version_dir}/${artifact_name}_ncu"
    ncu_report="${ncu_report_base}.ncu-rep"
    ncu_details="${version_dir}/${artifact_name}_ncu_details.txt"
    ncu_collection_log="${staging_dir}/.${version}.ncu-collection.log"

    if [[ ! -x "${source_executable}" ]]; then
        printf 'Missing executable: %s\n' "${source_executable}" >&2
        exit 1
    fi

    mkdir -p "${version_dir}"
    cp "${source_executable}" "${archived_executable}"

    printf '[%s] Running correctness and timing benchmark...\n' "${version}"
    {
        printf 'Collected at    : %s\n' "$(date --iso-8601=seconds)"
        printf 'Source version  : %s\n' "${version}"
        printf 'Executable      : %s\n\n' "${artifact_name}"
        "${archived_executable}"
    } >"${run_result}" 2>&1

    if ! grep -Eq 'Wrong results[[:space:]]*:[[:space:]]*0' "${run_result}"; then
        printf '[%s] Correctness validation failed.\n' "${version}" >&2
        tail -n 30 "${run_result}" >&2
        exit 1
    fi

    printf '[%s] Collecting NCU full details for %s...\n' \
        "${version}" "${kernels[${version}]}"
    if ! "${ncu_command[@]}" \
        --set full \
        --kernel-name-base function \
        --kernel-name "regex:${kernels[${version}]}" \
        --launch-count 1 \
        --import-source yes \
        --export "${ncu_report_base}" \
        --force-overwrite \
        "${archived_executable}" >"${ncu_collection_log}" 2>&1; then
        printf '[%s] Nsight Compute collection failed.\n' "${version}" >&2
        tail -n 40 "${ncu_collection_log}" >&2
        exit 1
    fi

    if grep -q 'ERR_NVGPUCTRPERM\|No kernels were profiled' \
        "${ncu_collection_log}"; then
        printf '[%s] Nsight Compute did not collect hardware counters.\n' \
            "${version}" >&2
        tail -n 40 "${ncu_collection_log}" >&2
        exit 1
    fi

    if [[ ! -s "${ncu_report}" ]]; then
        printf '[%s] Nsight Compute report was not created: %s\n' \
            "${version}" "${ncu_report}" >&2
        exit 1
    fi

    if ((use_sudo_ncu)); then
        sudo chown "$(id -u):$(id -g)" "${ncu_report}"
    fi

    if ! "${ncu_path}" \
        --import "${ncu_report}" \
        --page details \
        --print-details all \
        --print-rule-details \
        --print-summary per-kernel >"${ncu_details}" 2>&1; then
        printf '[%s] Failed to export NCU details text.\n' "${version}" >&2
        tail -n 40 "${ncu_details}" >&2
        exit 1
    fi

    rm -f "${ncu_collection_log}"
done

if [[ -e "${profiles_dir}" ]]; then
    mv "${profiles_dir}" "${backup_dir}"
fi

mv "${staging_dir}" "${profiles_dir}"
staging_dir=""
rm -rf "${backup_dir}"

printf '\nProfile archive created at %s\n' "${profiles_dir}"
printf 'Each version contains: executable, result.txt, ncu-rep, ncu_details.txt\n'
