total_frames=1_000_000_000
algorithm="mappo"  # mappo, maddpg, happo, mat, qmix
seed=0  # 0, 1, 2

CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=${total_frames} \
    task=MultiJuggleVolleyball \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.03 \
    eval_interval=50 \
    save_interval=50 \
    algo=${algorithm} \
    seed=${seed} \
    wandb.mode=disabled \


    