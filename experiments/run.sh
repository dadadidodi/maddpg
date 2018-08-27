#!/bin/bash
# python3 train.py --scenario simple_push  --num-adversaries 1 --save-dir './push_checkpoints_0/' --exp-name trial --adv-eps 0  --save-rate 1000 --num-episodes 10000 > push.log 2>&1 & 
# python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_0/' --exp-name trial_adversary --adv-eps 0  --save-rate 1000 --num-episodes 10000 > adversary.log 2>&1 & 
# python3 train.py --scenario simple_push  --num-adversaries 1 --save-dir './push_checkpoints_1/' --exp-name trial_logits --adv-eps 0  --save-rate 1000 --num-episodes 10000 > push_logits.log 2>&1 & 
# python3 train.py --scenario simple_push  --num-adversaries 1 --save-dir './push_checkpoints_2/' --exp-name trial_logits_adv --adv-eps 0  --save-rate 1000 --num-episodes 10000 > push_logits_adv.log 2>&1 & 
# python3 train.py --scenario simple_push  --num-adversaries 1 --save-dir './push_checkpoints_3/' --exp-name trial_logits_adv_pos --adv-eps 1e-3  --save-rate 100 --num-episodes 100 > push_logits_adv_e3.log 2>&1 & 
# python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_e3/' --exp-name adversary_e3 --adv-eps 1e-3  --save-rate 500 --num-episodes 10000 > adversary_e3.log 2>&1 & 
#python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_e9/' --exp-name adversary_e9 --adv-eps 1e-9  --save-rate 1000 --num-episodes 30000 > adversary_e9.log 2>&1 & 
#python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_e4/' --exp-name adversary_e4 --adv-eps 1e-4  --save-rate 1000 --num-episodes 30000 > adversary_e4.log 2>&1 & 
#python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_e5/' --exp-name adversary_e5 --adv-eps 1e-5  --save-rate 1000 --num-episodes 30000 > adversary_e5.log 2>&1 & 
#python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_e3/' --exp-name adversary_e3 --adv-eps 1e-3  --save-rate 1000 --num-episodes 30000 > adversary_e3.log 2>&1 & 
#python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_e2/' --exp-name adversary_e2 --adv-eps 1e-2  --save-rate 1000 --num-episodes 30000 > adversary_e2.log 2>&1 & 
#python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_e1/' --exp-name adversary_e1 --adv-eps 1e-1  --save-rate 1000 --num-episodes 30000 > adversary_e1.log 2>&1 & 
python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_5e1/' --exp-name adversary_5e1 --adv-eps 5e-1  --save-rate 1000 --num-episodes 30000 > adversary_5e1.log 2>&1 & 
python3 train.py --scenario simple_adversary  --num-adversaries 1 --save-dir './adversary_checkpoints_2e1/' --exp-name adversary_2e1 --adv-eps 2e-1  --save-rate 1000 --num-episodes 30000 > adversary_2e1.log 2>&1 & 
