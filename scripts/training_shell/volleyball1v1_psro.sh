CUDA_VISIBLE_DEVICES=0 python ../train_psro.py headless=true \
    total_frames=1_000_000_000 \
    task=Volleyball1v1 \
    task.drone_model=Iris \
    task.env.num_envs=2048 \
    task.env.env_spacing=18 \
    task.random_turn=True \
    task.initial.drone_0_near_side=True \
    task.symmetric_obs=True \
    solver_type=fsp \
    share_population=True \
    max_iter_steps=5000 \
    mean_threshold=0.90 \
    init_by_latest_strategy=True \
    seed=0 \
    wandb.mode=disabled # debug
    # solver_type: fsp, nash