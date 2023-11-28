#!/bin/bash
#SBATCH --partition=gpu3-2
#SBATCH --nodelist=g3030
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
cd /home/zhaoyue/CourseProject/IMAGECAPTION
conda activate course_env
python trycode.py
