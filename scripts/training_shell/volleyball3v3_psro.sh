CUDA_VISIBLE_DEVICES=0 python ../train_psro.py headless=true \
    total_frames=5_000_000_000 \
    task=Volleyball3v3 \
    task.drone_model=Iris \
    task.env.num_envs=2048 \
    task.env.env_spacing=30 \
    task.random_turn=True \
    task.symmetric_obs=True \
    solver_type=fsp \
    share_population=True \
    max_iter_steps=5000 \
    mean_threshold=0.90 \
    init_by_latest_strategy=True \
    max_population_size=10 \
    wandb.mode=disabled # debug
    # solver_type: fsp, nash
    
    
    