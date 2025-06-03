CUDA_VISIBLE_DEVICES=0 python ../train_psro.py headless=true \
    total_frames=1_000_000_000 \
    task=Volleyball6v6 \
    task.drone_model=Iris \
    task.env.num_envs=800 \
    task.env.env_spacing=50 \
    task.random_turn=True \
    task.symmetric_obs=True \
    share_population=True \
    max_iter_steps=5000 \
    mean_threshold=0.90 \
    solver_type=fsp \
    init_by_latest_strategy=True \
    max_population_size=10 \
    seed=0 \
    wandb.mode=disabled # debug
    # solver_type: fsp, nash
    
    
    