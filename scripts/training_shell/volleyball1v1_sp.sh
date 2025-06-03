CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=1_000_000_000 \
    task=Volleyball1v1 \
    task.drone_model=Iris \
    task.env.num_envs=2048 \
    task.env.env_spacing=18 \
    task.random_turn=True \
    task.initial.drone_0_near_side=True \
    task.symmetric_obs=True \
    eval_interval=300 \
    save_interval=300 \
    algo=mappo \
    algo.share_actor=true \
    algo.critic_input="obs" \
    seed=0 \
    wandb.mode=disabled # debug