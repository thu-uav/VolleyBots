CUDA_VISIBLE_DEVICES=0 python ../train_hier.py headless=true \
    total_frames=1_000_000_000 \
    task=Volleyball3v3_hier \
    task.drone_model=Iris \
    task.env.num_envs=1000 \
    task.env.env_spacing=30 \
    task.random_turn=True \
    task.symmetric_obs=True \
    eval_interval=300 \
    save_interval=300 \
    algo=mappo_hier \
    Opp_policy_checkpoint_path="crossplay/sp/0.pt" \
    FirstPass_serve_policy_checkpoint_path="hier/checkpoint_serve.pt" \
    FirstPass_serve_hover_policy_checkpoint_path="hier/checkpoint_serve_hover.pt" \
    FirstPass_goto_policy_checkpoint_path="hier/checkpoint_goto.pt" \
    FirstPass_policy_checkpoint_path="hier/checkpoint_firstpass.pt" \
    FirstPass_hover_policy_checkpoint_path="hier/checkpoint_firstpass_hover.pt" \
    SecPass_goto_policy_checkpoint_path="hier/checkpoint_goto.pt" \
    SecPass_policy_checkpoint_path="hier/checkpoint_secpass.pt" \
    SecPass_hover_policy_checkpoint_path="hier/checkpoint_secpass_hover.pt" \
    Att_goto_policy_checkpoint_path="hier/checkpoint_goto.pt" \
    Att_policy_checkpoint_path="hier/checkpoint_att.pt" \
    Att_hover_policy_checkpoint_path="hier/checkpoint_att_hover.pt" \
    seed=0 \
    run_name_suffix="eval_sp_seed_0" \
    only_eval=true \
    # wandb.mode=disabled \
    
    
    
    