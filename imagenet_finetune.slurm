#!/bin/bash
#SBATCH --job-name=imagenet_finetune    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
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
               --data_override AFHQ\
               --config afhq.yml      \
               --exp ./runs/test        \
               --edit_attr house_in_Nigeria      \
               --do_train 1             \
               --do_test 0 \
               --n_train_img 30         \
               --bs_train 2 \
               --n_iter 10               \
               --t_0 500                \
               --n_inv_step 40          \
               --lr_clip_finetune 8e-6  \
               --id_loss_w 0            \
               --clip_loss_w 5 \
               --n_train_step 10 \
               --n_precomp_img 30 \
               --model_path pretrained/512x512_diffusion.pt \
               --model_save_name imagenet_house_finetune \
		       -—finetune_class_name house



