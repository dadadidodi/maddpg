#!/bin/bash

declare -a advarr=( 'mmmaddpg')
declare -a norarr=('mmmaddpg')
declare -A arr
arr+=(["mmmaddpg"]="../tag_checkpoints_e12/model-145000")
arr+=(["maddpg"]="../tag_checkpoints_maddpg/model-100000")
for adv in "${advarr[@]}"
do
    for nor in "${norarr[@]}"
    do
        echo "adv $adv normal $nor"
        echo ${arr[${adv}]}
        echo ${arr[${nor}]}
        python3 ../train_seperate.py --scenario simple_tag --num-adversaries 3 --good-policy "$norarr" --bad-policy "$advarr" --load-good ${arr[${nor}]} --load-bad ${arr[${adv}]} --test --adv-eps 0 --adv-eps-s 0 --num-episodes 2500 --save-rate 2500 --exp-name test_adv &
    done
done

