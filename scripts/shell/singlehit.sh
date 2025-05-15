total_frames=500_000_000
share_actor=true    # true, false
algorithm="mappo"  # mappo, maddpg, td3, sac, dqn
action_transform="PIDrate"  # PIDrate, null
throttles_in_obs=false # true, false
wandb_project="omnidrones"
seed=0  # 0, 1, 2

CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=${total_frames} \
    task=SingleHitVolleyball \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    task.ball_offset="[0., 0., 3.]" \
    eval_interval=50 \
    save_interval=500 \
    algo=${algorithm} \
    task.time_encoding=false \
    task.action_transform=${action_transform}\
    task.throttles_in_obs=${throttles_in_obs}\
    seed=${seed} \
    wandb.project=${wandb_project} \
    # wandb.mode=disabled \