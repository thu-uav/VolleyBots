CUDA_VISIBLE_DEVICES=0 python ../train_psro_cross_play.py headless=true \
    total_frames=1_000_000_000 \
    task=Volleyball3v3 \
    task.drone_model=Iris \
    task.env.num_envs=1 \
    task.env.env_spacing=30 \
    task.random_turn=True \
    task.symmetric_obs=True \
    run_name_suffix="psro_uniform_vs_psro_nash" \
    seed=0 \
    append_actors_0_from_path="../../checkpoints/crossplay/psro_uniform_population" \
    append_actors_1_from_path="../../checkpoints/crossplay/psro_nash_population" \
    load_meta_policy_1_path="../../checkpoints/crossplay/psro_nash_meta_policy.npz" \
    wandb.mode=disabled \
    # only_eval=true \
    
    
    