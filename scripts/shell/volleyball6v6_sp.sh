CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=1_000_000_000 \
    task=Volleyball6v6 \
    task.drone_model=Iris \
    task.env.num_envs=800 \
    task.env.env_spacing=50 \
    task.random_turn=True \
    task.symmetric_obs=True \
    eval_interval=1000 \
    save_interval=1000 \
    algo=mappo \
    algo.share_actor=true \
    algo.critic_input="obs" \
    seed=0 \
    run_name_suffix="seed_0" \
    # wandb.mode=disabled # debug
    # only_eval=true \
    # policy_checkpoint_path="checkpoint/checkpoint_786497536.pt"\
    
    
    