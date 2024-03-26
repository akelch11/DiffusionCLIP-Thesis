#!/bin/bash
#SBATCH --job-name=geode_interpolate_latents_Americas_spices  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=akelch@princeton.edu

#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu80
#SBATCH --mem=15G

module purge
module load anaconda3/2023.9
conda activate torch-thesis

 ! python main.py --interpolate_latents                            --data_override GEODE                    --config afhq.yml                          --exp ./runs/test                            --edit_attr spices_in_Americas                          --do_train 1                                 --do_test 0                     --n_iter 1                                  --t_0 500                                    --n_inv_step 40                         --n_train_step 10                     --n_test_step 40                     --model_path checkpoint/geode_spices_Americas_finetune_NORMAL_30_ddim_l1_10.0-9.pt                    --model_save_name geode_spices_Americas_finetune                     --finetune_class_name spices                     --finetune_region Americas                     --lambda_step 0.25                     --latent_mult 5