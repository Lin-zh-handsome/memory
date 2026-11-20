#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
tasks=(observe_and_pickup rearrange_blocks swap_T blocks_ranking_try press_button)
status="$ROOT/logs/download_status.json"
publish_status() {
    local temp="$status.tmp"
    local start
    start=$(awk '{print $22}' "/proc/$$/stat")
    printf '{"finished":%s,"pid":%s,"start":"%s"}\n' "$1" "$$" "$start" > "$temp" && mv "$temp" "$status"
}
publish_status false
failed=0
for task in "${tasks[@]}"; do
    url_path="datasets/TianxingChen/RMBench/resolve/main/data/$task/demo_clean.zip"
    path="$ROOT/datasets/$task.zip"
    [[ -s "$path" ]] && continue
    for domain in hf-mirror.com huggingface.co; do
        if curl -fsSL --retry 2 --connect-timeout 15 --max-time 900 -C - -o "$path.part" "https://$domain/$url_path"; then
            mv "$path.part" "$path"
            break
        fi
    done
    if [[ -s "$path" ]]; then
        echo "downloaded $task"
    else
        echo "unavailable $task" >&2
        failed=1
    fi
done
publish_status true
exit "$failed"
