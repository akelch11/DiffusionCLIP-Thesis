#!/bin/bash
#SBATCH --job-name=geode_generate_synth_hand_soap_SouthEastAsia_NORMAL_b0.2    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=07:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=akelch@princeton.edu
#SBATCH --gpus-per-node=1
#SBATCH --mem=30G

module purge
module load anaconda3/2023.9
conda activate torch-thesis

! python main.py --generate_synth                --data_override GEODE                --config afhq.yml                   --exp ./runs/test                        --do_train 1                             --do_test 0                 --n_iter 1                              --t_0 500                                --n_inv_step 40                     --n_train_step 10                 --n_test_step 40                 --model_path checkpoint/geode_hand_soap_SouthEastAsia_finetune_NORMAL_30_ddim_l1_10.0-9.pt                 --latent_file_path geode_hand_soap_SouthEastAsia_finetune                 --latent_mult 5                 --bandwidth 0.2