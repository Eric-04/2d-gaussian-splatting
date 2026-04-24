import os
import shutil
import tempfile
from huggingface_hub import snapshot_download, list_repo_files

repo_id = "GaussianWorld/dl3dv_10k_3dgs_new"

# Find scene IDs
files = list_repo_files(repo_id, repo_type="dataset")
scene_ids = sorted(set(f.split("/")[1] for f in files if f.startswith("10K/")))

# Take first N
scene_ids = scene_ids[:2]

for i, sid in enumerate(scene_ids):
    final_path = f"./output/dl3dv/{sid}"

    # Skip if already downloaded
    if os.path.exists(final_path):
        print(f"Skipping {sid}")
        continue

    with tempfile.TemporaryDirectory() as tmp_dir:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=tmp_dir,
            allow_patterns=f"10K/{sid}/*",
            local_dir_use_symlinks=False
        )

        src = os.path.join(tmp_dir, "10K", sid)

        # Move the folder contents to final destination
        shutil.move(src, final_path)