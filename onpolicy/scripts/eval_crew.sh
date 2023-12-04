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
    CUDA_VISIBLE_DEVICES=0 python eval/eval_crew.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --crew_name ${crew} --num_agents ${num_agents} --num_tasks ${num_tasks} --num_hints ${num_hints} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 100 --ppo_epoch 15 --gamma 1 --log 1 \
    --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 --user_name=qhughes22 --run_name hint_test --use_eval --use_wandb \
    --model_dir "/home/haoming/extreme_driving/Adeesh/MLGT/marl-the-crew/onpolicy/scripts/results/The_Crew/TheCrew-small-no_rocket/mappo/check/wandb/run-20231203_195852-3sjhfzbd/files"
done
echo "Eval is done!"
