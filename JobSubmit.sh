#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -p qTRDGPUH
#SBATCH -t 7000
#SBATCH -D /data/users2/yxiao11/model/pix_con
#SBATCH -J 50slice
#SBATCH -e 50slice.err
#SBATCH -o 50slice.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yxiao11@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --gres=gpu:v100:4

sleep 3s

source /data/users2/yxiao11/.bashrc
source activate p37


# python main_2d.py -log_path logs_add_64bits -num_colors 64 -d Addiction -epochs 2000 -depth_rs 29 -num_filters_cond 72
# python demo.py -log_path logs_add_64bits -num_colors 64 -d Addiction BSNIP ABCD COBRE -depth_rs 29 -num_filters_cond 144 -num_filters_prior 144

# python main_2d.py -log_path logs_add_mri_deface -num_colors 8 -d Add_mri_deface -epochs 500 -num_filters_cond 72

#-------------------for 3d training------------------------------------

python main_3d.py -log_path logs_add_50_slices -start_slice 97 -d Addiction -epochs 300 -num_filters_cond 72 -num_slices 50
# python demo_3d.py -log_path logs_add_3d -start_slice 97 -num_slices 50 -num_filters_cond 72

sleep 10s     