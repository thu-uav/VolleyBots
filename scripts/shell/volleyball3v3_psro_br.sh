CUDA_VISIBLE_DEVICES=0 python ../train_psro_br.py headless=true \
    total_frames=1_000_000_000 \
    task=Volleyball3v3 \
    task.drone_model=Iris \
    task.env.num_envs=2048 \
    task.env.env_spacing=30 \
    task.random_turn=True \
    task.symmetric_obs=True \
    solver_type=fsp \
    max_iter_steps=5000 \
    mean_threshold=1.00 \
    seed=0 \
    append_actors_0_from_path="/home/zhangruize/OmniDrones/scripts/shell/br/population" \
    # wandb.mode=disabled # debug
    # load_meta_policy_path="/home/zhangruize/OmniDrones/scripts/shell/br/meta_policy_iter_2.npz" \
    
    
    