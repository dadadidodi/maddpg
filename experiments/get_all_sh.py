#!/usr/bin/env python
# encoding: utf-8

import argparse

parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

parser.add_argument("-s", "--scenario", choices=['adv', 'crypto', 'push', 'tag'], help="choices of scenario")

parser.add_argument("-l", "--locations", nargs='+', help="locations")
# parser.add_argument("-bm", "--bad-method", default="ma", choices=['ma', 'm3', 'ma_ens', 'm3_ens'], help="method")
# parser.add_argument("-lb", "--load-bad", type=str, default=None)
# parser.add_argument("-gm", "--good-method", default="ma", choices=['ma', 'm3', 'ma_ens', 'm3_ens'], help="method")
# parser.add_argument("-lg", "--load-good", type=str, default=None)

parser.add_argument("-ex", "--extra_info", type=str, default=None)
# parser.add_argument("-t", "--test", action='store_true')

args = parser.parse_args()
args.num_ens = 3
if args.scenario == 'adv':
    args.num_adv = 1
    args.env_name = 'simple_adversary'
elif args.scenario == 'crypto':
    args.num_adv = 1
    args.env_name = 'simple_crypto'
elif args.scenario == 'push':
    args.num_adv = 2
    args.env_name = 'simple_push'
elif args.scenario == 'tag':
    args.num_adv = 3
    args.env_name = 'simple_tag'
    args.num_ens = 2

def get_cmd(bad_method, load_bad, good_method, load_good, bad_extra_info=None, good_extra_info=None):
    cmd = 'python3 train_seperate.py'

    names = [args.scenario, bad_method]
    if bad_extra_info is not None:
        names.append(bad_extra_info)
    names.append('vs')
    names.append(good_method)
    if good_extra_info is not None:
        names.append(good_extra_info)
    exp_name = '_'.join(names)

    cmd += ' --scenario {}'.format(args.env_name)
    cmd += ' --num-adversaries {}'.format(args.num_adv)
    cmd += ' --save-dir ./{} --exp-name {}'.format(exp_name, exp_name)

    policy_name = dict(ma='maddpg', m3='mmmaddpg', ma_ens='maddpg', m3_ens='mmmaddpg')
    cmd += ' --bad-policy {}'.format(policy_name[bad_method])
    if load_bad is not None:
        cmd += ' --load-bad {}'.format(load_bad)
    if bad_method.endswith('ens'):
        cmd += ' --bad-ensemble {}'.format(args.num_ens)

    cmd += ' --good-policy {}'.format(policy_name[good_method])
    if load_good is not None:
        cmd += ' --load-good {}'.format(load_good)
    if good_method.endswith('ens'):
        cmd += ' --good-ensemble {}'.format(args.num_ens)

    cmd += ' --save-rate 10000 --num-episodes 10000 --test'

    cmd += ' > {}.log'.format(exp_name)
    return cmd

def main():
    methods = ['ma', 'm3', 'ma_ens', 'm3_ens']
    assert len(args.locations) == len(methods)
    all_cmd = ''

    for bm, lb in zip(methods, args.locations):
        for gm, lg in zip(methods, args.locations):
            bad_extra_info = good_extra_info = None
            if bm == 'm3_ens':
                bad_extra_info = args.extra_info
            if gm == 'm3_ens':
                good_extra_info = args.extra_info
            cmd = get_cmd(bm, lb, gm, lg, bad_extra_info, good_extra_info)
            print(cmd)
            if all_cmd != '':
                all_cmd += ' & '
            all_cmd += cmd
    print(all_cmd)

if __name__  == "__main__":
    main()
