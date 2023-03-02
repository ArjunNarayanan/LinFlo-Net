#!/bin/bash
# Job name:
#SBATCH --job-name=eval-model-5-mr
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio2
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks-per-node=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=00:20:00
#
## Command(s) to run (example):


config_file=config/segment_flow/model-5/evaluate_mr.yml
python utilities/evaluate_meshing_model.py -config $config_file
