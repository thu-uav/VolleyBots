total_frames=1_000_000_000  # 1_000_000_000 for results, 500_000_000 for debug
share_actor=true    # true, false
reward_shaping=true  # true, false
algorithm="mappo"  # mappo, maddpg, happo, mat
wandb_project="omnidrones"
seed=0  # 0, 1, 2

CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=${total_frames} \
    task=MultiAttackVolleyballEasy \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    task.reward_shaping=${reward_shaping} \
    viewer.eye="[8., 0., 5.]" \
    viewer.lookat="[-4., 0., 0.]" \
    eval_interval=50 \
    save_interval=500 \
    algo=${algorithm} \
    algo.share_actor=${share_actor}  \
    task.time_encoding=false \
    seed=${seed} \
    wandb.project=${wandb_project} \
    # wandb.mode=disabled \