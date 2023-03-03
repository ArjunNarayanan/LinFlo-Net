srun --pty -A fc_biome -p savio2_1080ti --nodes=1 --gres=gpu:1 --ntasks=1 --cpus-per-task=2 -t 00:30:00 bash -i


srun --pty -A fc_biome -p savio3_gpu --nodes=1 --gres=gpu:GTX2080TI:1 --ntasks=1 --cpus-per-task=2 -t 00:30:00 bash -i

