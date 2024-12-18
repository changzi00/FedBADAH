#!/bin/bash

python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0

python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 10 --epochs 32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0

python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0

python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 50 --epochs 32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0

python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0

python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
python main_fedbadah.py --dataset cifar100 --model mobile --num_classes 100 --shard_per_user 100 --epochs 32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0
