total_frames=500_000_000
algorithm="mappo"  # mappo, maddpg, td3, sac, dqn
action_transform=""  # PIDrate, null
throttles_in_obs=true # true, false
wandb_project="omnidrones"
seed=0  # 0, 1, 2

CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=${total_frames} \
    task=BackAndForth \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    viewer.eye="[16., 0., 8.]" \
    viewer.lookat="[0., 0., 0.]" \
    eval_interval=50 \
    save_interval=500 \
    algo=${algorithm} \
    task.time_encoding=false \
    task.action_transform=${action_transform}\
    task.throttles_in_obs=${throttles_in_obs}\
    seed=${seed} \
    wandb.mode=disabled \
    # wandb.project=${wandb_project} \