#!/usr/bin/env bash

set -euo pipefail

runs="${1:-5}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build/sm_89}"

if [[ "${build_dir}" != /* ]]; then
    build_dir="${repo_dir}/${build_dir}"
fi

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

if ! [[ "$runs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'RUNS must be a positive integer, got: %s\n' "$runs" >&2
    exit 2
fi

printf '| Version | Mean time | Mean effective BW | Correctness |\n'
printf '|---|---:|---:|---:|\n'

for version in "${versions[@]}"; do
    executable="${build_dir}/${version}"

    if [[ ! -x "$executable" ]]; then
        printf 'Missing executable: %s\n' "$executable" >&2
        printf 'Run: make all\n' >&2
        exit 1
    fi

    times=()
    bandwidths=()

    for ((run = 1; run <= runs; ++run)); do
        output="$($executable)"
        time_ms="$(awk -F: '/Average time/ {gsub(/[^0-9.]/, "", $2); print $2; exit}' <<<"$output")"
        bandwidth="$(awk -F: '/Effective BW/ {gsub(/[^0-9.]/, "", $2); print $2; exit}' <<<"$output")"
        wrong="$(awk -F: '/Wrong results/ {gsub(/[^0-9]/, "", $2); print $2; exit}' <<<"$output")"

        if [[ -z "$time_ms" || -z "$bandwidth" || -z "$wrong" ]]; then
            printf 'Could not parse benchmark output for %s (run %d).\n' "$version" "$run" >&2
            exit 1
        fi

        if [[ "$wrong" != "0" ]]; then
            printf '%s failed validation with %s wrong results.\n' "$version" "$wrong" >&2
            exit 1
        fi

        times+=("$time_ms")
        bandwidths+=("$bandwidth")
    done

    mean_time="$(printf '%s\n' "${times[@]}" | awk '{sum += $1} END {printf "%.6f", sum / NR}')"
    mean_bandwidth="$(printf '%s\n' "${bandwidths[@]}" | awk '{sum += $1} END {printf "%.3f", sum / NR}')"

    printf '| `%s` | %s ms | %s GB/s | pass |\n' \
        "$version" "$mean_time" "$mean_bandwidth"
done
