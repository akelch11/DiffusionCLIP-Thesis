#!/bin/bash
#SBATCH --job-name=geode_finetune_small_test   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=akelch@princeton.edu

#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu80
#SBATCH --mem=80G



module purge
module load anaconda3/2023.9
conda activate torch-thesis


! python main.py --clip_finetune_eff        \
               --data_override GEODE\
               --config afhq.yml      \
               --exp ./runs/test        \
               --edit_attr house_in_Nigeria      \
               --do_train 1             \
               --do_test 0 \
               --n_train_img 5         \
               --bs_train 1 \
               --n_iter 2               \
               --t_0 100                \
               --n_inv_step 40          \
               --lr_clip_finetune 8e-6  \
               --id_loss_w 10            \
               --clip_loss_w 0 \
               --n_train_step 2 \
               --n_precomp_img 5 \
               --save_train_image 0 \
               --model_path pretrained/512x512_diffusion.pt \
               --model_save_name geode_house_finetune \
               --finetune_class_name house \
               --finetune_region Nigeria
               



