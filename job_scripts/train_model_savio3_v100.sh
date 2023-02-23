#!/bin/bash
# Job name:
#SBATCH --job-name=Train-LT-Seg-Flow-model-3
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# QoS:
#SBATCH --qos=v100_gpu3_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks-per-node=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=4
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:V100:1
#
# Wall clock limit:
#SBATCH --time=70:00:00
#
## Command(s) to run (example):


config_file=output/segment_flow/model-3/config.yml
python train_segment_flow.py -config $config_file
