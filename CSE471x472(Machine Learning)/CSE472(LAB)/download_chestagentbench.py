from huggingface_hub import snapshot_download

print("Starting download...")

snapshot_download(
    repo_id="wanglab/chest-agent-bench",
    repo_type="dataset",
    local_dir="chestagentbench",
    local_dir_use_symlinks=False
)

print("Download complete!")
