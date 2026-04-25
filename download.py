import os
import shutil
import time
import tempfile
from huggingface_hub import snapshot_download, list_repo_files

repo_id = "DL3DV/DL3DV-Benchmark"

# Find scene IDs (top-level dirs)
files = list_repo_files(repo_id, repo_type="dataset")
scene_ids = sorted(set(f.split("/")[0] for f in files if "/" in f if not f.startswith(".")))

# Take first N
scene_ids = scene_ids[:10]
folders = ["images", "images_2", "images_4", "images_8", "sparse"]

for sid in scene_ids:
    final_path = f"./data/dl3dv/{sid}"

    if os.path.exists(final_path):
        print(f"Skipping {sid}")
        continue

    os.makedirs(final_path, exist_ok=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for folder in folders:
            start_time = time.time()

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=tmp_dir,
                allow_patterns=[f"{sid}/gaussian_splat/{folder}/*"],
                local_dir_use_symlinks=False
            )

            print(f"Finished {folder}, waiting 5 minutes...")
            remaining = 300 - (time.time() - start_time)
            if folder != "sparse" and remaining > 0:
                time.sleep(remaining)

        base_src = os.path.join(tmp_dir, sid, "gaussian_splat")

        for folder in folders:
            src_folder = os.path.join(base_src, folder)
            dst_folder = os.path.join(final_path, folder)

            if os.path.exists(src_folder):
                shutil.move(src_folder, dst_folder)

    print(f"Downloaded {sid}")