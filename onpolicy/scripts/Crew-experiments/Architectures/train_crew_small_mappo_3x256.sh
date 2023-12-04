#!/bin/sh
env="The_Crew"
crew="TheCrew-small-no_rocket"
num_agents=4
algo="mappo"
exp="check"
num_tasks=1
num_hints=1
seed_max=1
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ./../../train/train_crew.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --crew_name ${crew} --num_agents ${num_agents} --num_tasks ${num_tasks} --num_hints ${num_hints} --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 100000 --ppo_epoch 15 --gamma 1 \
    --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 256 --layer_N 3 --entropy_coef 0.015 --user_name=qhughes22 --run_name crew-small-mappo-2x512
    echo "training is done!"
done
