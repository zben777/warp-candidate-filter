#!/usr/bin/env bash

set -euo pipefail


usage()
{
    cat <<'EOF'
Usage:
    ./scripts/collect_profiles.sh [--sudo-ncu]
EOF
}


use_sudo_ncu=0

case "${1:-}" in
    "")
        ;;
    --sudo-ncu)
        use_sudo_ncu=1
        ;;
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
    if [[ -d "${staging_dir}" ]]; then
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
    v9
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
    [v9]=v9_subwarp_cp_before_a_kernel
)



ncu_path="$(command -v ncu || true)"

if [[ -z "${ncu_path}" ]]; then
    echo "ERROR: ncu not found"
    exit 1
fi



if ((use_sudo_ncu)); then

    sudo -v

    ncu_command=(sudo "${ncu_path}")

else

    ncu_command=("${ncu_path}")

fi



echo "Building all sm_89 executables..."

make -C "${repo_dir}" all



for version in "${versions[@]}"; do


    echo
    echo "=============================="
    echo "Profiling ${version}"
    echo "=============================="


    source_executable="${repo_dir}/build/sm_89/${version}"


    if [[ ! -x "${source_executable}" ]]; then
        echo "Missing executable:"
        echo "${source_executable}"
        exit 1
    fi



    version_dir="${staging_dir}/${version}"

    mkdir -p "${version_dir}"



    artifact_name="${version}_real_4b"

    archived_executable="${version_dir}/${artifact_name}"



    run_result="${version_dir}/${artifact_name}_result.txt"



    ncu_report_base="${version_dir}/${artifact_name}_ncu"

    ncu_report="${ncu_report_base}.ncu-rep"



    ncu_details="${version_dir}/${artifact_name}_ncu_details.txt"

    ncu_source_sass="${version_dir}/${artifact_name}_source_sass.txt"

    sass_file="${version_dir}/${artifact_name}.sass"



    cp "${source_executable}" "${archived_executable}"



    echo "[${version}] Running benchmark..."

    {
        echo "Collected:"
        date --iso-8601=seconds

        echo
        echo "Version:"
        echo "${version}"

        echo
        echo "Executable:"
        echo "${artifact_name}"

        echo

        "${archived_executable}"

    } > "${run_result}" 2>&1



    if ! grep -Eq \
        'Wrong results[[:space:]]*:[[:space:]]*0' \
        "${run_result}";
    then

        echo "[${version}] Correctness failed"

        tail -n 30 "${run_result}"

        exit 1

    fi



    echo "[${version}] Collecting NCU..."



    "${ncu_command[@]}" \
        --set full \
        --kernel-name-base function \
        --kernel-name "regex:${kernels[${version}]}" \
        --launch-skip 10 \
        --launch-count 1 \
        --import-source yes \
        --export "${ncu_report_base}" \
        --force-overwrite \
        "${archived_executable}"



    if [[ ! -s "${ncu_report}" ]]; then

        echo "NCU report missing:"
        echo "${ncu_report}"

        exit 1

    fi



    if ((use_sudo_ncu)); then

        sudo chown \
            "$(id -u):$(id -g)" \
            "${ncu_report}"

    fi



    echo "[${version}] Export details..."

    "${ncu_path}" \
        --import "${ncu_report}" \
        --page details \
        > "${ncu_details}"



    echo "[${version}] Export CUDA + SASS mapping..."

    "${ncu_path}" \
        --import "${ncu_report}" \
        --page source \
        --print-source cuda,sass \
        > "${ncu_source_sass}"



    echo "[${version}] Dump SASS..."

    cuobjdump \
        --dump-sass "${archived_executable}" \
        > "${sass_file}"



    echo "[${version}] Done"


done



if [[ -e "${profiles_dir}" ]]; then
    mv "${profiles_dir}" "${backup_dir}"
fi



mv "${staging_dir}" "${profiles_dir}"

staging_dir=""



rm -rf "${backup_dir}"

trap - EXIT



echo
echo "================================"
echo "Profiles created:"
echo "${profiles_dir}"
echo "================================"
