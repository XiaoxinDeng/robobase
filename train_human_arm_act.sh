python train.py method=act launch=act_pixel_bigym env=bigym/drawer_top_open.yaml wandb.name=drawer_top_open
python external/robobase/train.py method=act launch=act_pixel_bigym env=bigym/drawer_top_open.yaml


python train.py method=act launch=act_pixel_bigym env=bigym/human_arm_drawer_top_open.yaml
python external/robobase/train.py method=act launch=act_pixel_bigym env=bigym/human_arm_drawer_top_open.yaml