cd ~/Workspace/safe_bigym_hoi
git pull
git switch dev
cd external/robobase
git pull
git reset --hard
git switch dev
# Train ACT on BigYM drawer_top_open environment
tmux new -d -s Train 'cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase/ && python3 train.py method=act env=bigym/drawer_top_open.yaml launch=act_pixel_bigym 2>&1 | tee -a /home/xd1125/Workspace/safe_bigym_hoi/external/robobase//train.log'
# Continue training from a checkpoint
tmux new -d -s Train 'cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase/ && python3 resume_train.py method=act env=bigym/drawer_top_open.yaml launch=act_pixel_bigym 2>&1 | tee -a /home/xd1125/Workspace/safe_bigym_hoi/external/robobase//train.log'

python3 train.py method=act env=bigym/drawer_top_open.yaml launch=act_pixel_bigym
#  env.episode_length=100 \
#  demos=10 pixels=true