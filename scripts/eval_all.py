import subprocess
import os

scenes = ["SCENE1", "SCENE2"]
gt_poses = r"H:\Documents\calib-board\RUN1_n8_no_drift\split\SCENE12\1\test\camera_poses.json"
base_path = r".\RUN1_n8_all"

if __name__ == "__main__":

    for scene in scenes:
        path = os.path.join(base_path, scene)
        pxs = sorted(os.listdir(path))
        for px in pxs:
            est_poses = os.path.join(path, px, "camera_poses.json")
            subfolders = [s for s in os.listdir(os.path.join(path, px)) if os.path.isdir(os.path.join(path, px, s)) if s != "frames"]

            if os.path.exists(est_poses):
                subprocess.call([
                    "H:\\Documents\\calib-commons\\.calib\\Scripts\\python.exe",
                    r".\scripts\eval_calib.py",
                    "--path", f"{est_poses}",
                    "--gt_path", f"{gt_poses}",
                    "--scale", "--save_json"
                ])

            if subfolders:
                for subfolder in subfolders:
                    est_poses = os.path.join(path, px, subfolder, "camera_poses.json")
                    if os.path.exists(est_poses):
                        subprocess.call([
                            "H:\\Documents\\calib-commons\\.calib\\Scripts\\python.exe",
                            r".\scripts\eval_calib.py",
                            "--path", f"{est_poses}",
                            "--gt_path", f"{gt_poses}",
                            "--scale", "--save_json"
                        ])