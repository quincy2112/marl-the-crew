#!/bin/sh
env="The_Crew"
crew="TheCrew-standard"
num_agents=4
algo="mappo"
exp="check"
num_tasks=1
num_hints=1
seed_max=1
ulimit -n 22222
# WANDB_ENTITY="qhughes22/the_crew"
# WANDB_PROJECT="MAPPO-Crew"

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_crew.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --crew_name ${crew} --num_agents ${num_agents} --num_tasks ${num_tasks} --num_hints ${num_hints} --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 --gamma 1 \
    --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 --run_name full_game
    echo "training is done!"
done
