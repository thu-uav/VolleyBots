CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=1_000_000_000 \
    task=MultiJuggleVolleyball \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.03 \
    eval_interval=50 \
    save_interval=50 \
    algo=mappo \
    # wandb.mode=disabled # debug
    #algo.share_actor=true \
    #algo.critic_input="obs" \
    #task.time_encoding=false \

    