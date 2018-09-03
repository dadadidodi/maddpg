#!/bin/bash
python3 train_seperate.py --scenario simple_push  --num-adversaries 2 --save-dir './push_checkpoints_e11/' --exp-name push_e11 --adv-eps 1e-1  --adv-eps-s 1e-1   --good-policy mmmaddpg --adv-policy mmmaddpg --save-rate 2500 --num-episodes 150000 
python3 train_seperate.py --scenario simple_push  --num-adversaries 2 --save-dir './push_checkpoints_e12/' --exp-name push_e12 --adv-eps 1e-1  --adv-eps-s 1e-2  --good-policy mmmaddpg --adv-policy mmmaddpg --save-rate 2500 --num-episodes 150000 
python3 train_seperate.py --scenario simple_push  --num-adversaries 2 --save-dir './push_checkpoints_e23/' --exp-name push_e23 --adv-eps 1e-2  --adv-eps-s 1e-3  --good-policy mmmaddpg --adv-policy mmmaddpg --save-rate 2500 --num-episodes 150000
python3 train_seperate.py --scenario simple_push --num-adversaries 2 --save-dir './push_checkpoints_maddpg/' --exp-name push_maddpg --adv-eps 0 --adv-eps-s 0 --good-policy maddpg --adv-policy maddpg --save-rate 2500 --num-episodes 150000 
