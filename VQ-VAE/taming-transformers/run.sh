#!/bin/bash
#$ -N train
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
source /scratch/acb11361bd/StoryGan/VIST_VAE/work/bin/activate
module load gcc/9.3.0
module load python/3.8/3.8.7
module load cuda/11.1/11.1.1


python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1,2,3
#qsub -g gcb50169 -l RESOURCE_TYPE=NUM_RESOURCE BATCH_FILE
