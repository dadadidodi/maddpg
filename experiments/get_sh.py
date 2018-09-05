#!/usr/bin/env python
# encoding: utf-8

import argparse

parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

parser.add_argument("-s", "--scenario", choices=['adv', 'crypto', 'push', 'tag'], help="choices of scenario")
parser.add_argument("-bm", "--bad-method", default="ma", choices=['ma', 'm3', 'ma_ens', 'm3_ens'], help="method")
parser.add_argument("-bl", "--bad-load", type=str, default=None)

parser.add_argument("-gm", "--good-method", default="ma", choices=['ma', 'm3', 'ma_ens', 'm3_ens'], help="method")
parser.add_argument("-gl", "--good-load", type=str, default=None)

parser.add_argument("-ex", "--extra_info", type=str, default=None)
parser.add_argument("-t", "--test", action='store_true')

args = parser.parse_args()

policy_name = dict(ma='maddpg', m3='mmmaddpg', ma_ens='maddpg', m3_ens='mmmaddpg')

cmd = 'python3 train_seperate.py'

num_ens = 3
if args.scenario == 'adv':
    num_adv = 1
    env_name = 'simple_adversary'
elif args.scenario == 'crypto':
    num_adv = 1
    env_name = 'simple_crypto'
elif args.scenario == 'push':
    num_adv = 2
    env_name = 'simple_push'
elif args.scenario == 'tag':
    num_adv = 3
    env_name = 'simple_tag'
    num_ens = 2

names = [args.scenario, args.bad_method, 'vs', args.good_method]
if args.extra_info is not None:
    names.append(args.extra_info)
exp_name = '_'.join(names)

cmd += ' --scenario {}'.format(env_name)
cmd += ' --num-adversaries {}'.format(num_adv)
cmd += ' --save-dir ./{} --exp-name {}'.format(exp_name, exp_name)

cmd += ' --bad-policy {}'.format(policy_name[args.bad_method])
if args.bad_load is not None:
    cmd += ' --bad-load {}'.format(args.bad_load)
if args.bad_method.endswith('ens'):
    cmd += ' --bad-ensemble {}'.format(num_ens)

cmd += ' --good-policy {}'.format(policy_name[args.good_method])
if args.good_load is not None:
    cmd += ' --good-load {}'.format(args.good_load)
if args.good_method.endswith('ens'):
    cmd += ' --good-ensemble {}'.format(num_ens)

cmd += ' --save-rate 2500 --num-episodes 10000'

if args.test:
    cmd += ' --test'

cmd += ' > {}.log'.format(exp_name)

print(cmd)