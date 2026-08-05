
from robobase.workspace import Workspace   # or from robobase.train import Workspace, depending on your file layout
from pathlib import Path
import hydra
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


@hydra.main(config_path="robobase/cfgs", config_name="robobase_config", version_base=None)
def main(cfg):
    snapshot_path = Path("/home/xd1125/Workspace/safe_bigym_hoi/exp_local/pixel_act/bigym_human_arm_cupboards_open_all_20260304142316/snapshots/latest_snapshot.pt")
    # run dir = snapshot_dir/..
    run_dir = snapshot_path.parent.parent
    
    ws = Workspace(cfg, work_dir=run_dir)
    print("work_dir:", ws.work_dir)
    print("eval_video_dir:", Path(ws.work_dir) / "eval_videos")
    print("log_eval_video:", cfg.log_eval_video)

    ws.load_snapshot(snapshot_path)
    metrics = ws.eval()
    for key, value in metrics.items():
        if key != "eval_rollout": print(f"{key}: {value}")

    video_dir = Path(ws.work_dir) / "eval_videos"
    print("video_dir exists:", video_dir.exists())
    if video_dir.exists():
        print("saved videos:", [str(p) for p in video_dir.glob("*.mp4")])


if __name__ == "__main__":
    main()