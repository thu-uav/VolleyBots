total_frames=500_000_000
algorithm="maddpg"  # mappo, maddpg, td3, sac, dqn
action_transform=""  # PIDrate, null
seed=0  # 0, 1, 2

CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=${total_frames} \
    task=Hover \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    viewer.eye="[16., 0., 8.]" \
    viewer.lookat="[0., 0., 0.]" \
    eval_interval=50 \
    save_interval=500 \
    algo=${algorithm} \
    task.time_encoding=false \
    task.action_transform=${action_transform}\
    seed=${seed} \
    wandb.mode=online \
    wandb.entity=nics_marl \
    wandb.project=omnidrones \
    run_name_suffix="seed_0" \