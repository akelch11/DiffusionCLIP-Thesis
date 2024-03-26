



import os

boilerplate = f'''#!/bin/bash

#SBATCH --job-name=geode_generate_synth_house    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=25:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=akelch@princeton.edu

#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu80
#SBATCH --mem=30G



module purge
module load anaconda3/2023.9
conda activate torch-thesis
'''

classes =  [
            'house', 
            'spices', 
            'religious_building', 
            'hand_soap',
            'dustbin',
            "medicine",
            "car", 
            "storefront",
            "plate_of_food", 
            ]
regions = ["Africa", "Americas", "EastAsia", "Europe", "SouthEastAsia", "WestAsia"]


for class_name in classes:
    for region in regions:
        for p_set in ["NORMAL", "BIG"]:
            for b in [0.0, 0.2]:

        

                boilerplate = f'''#!/bin/bash
#SBATCH --job-name=geode_generate_synth_{class_name}_{region}_{p_set}_b{b}    # create a short name for your job
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
'''
            
                if p_set == 'BIG':
                    model_name = f'geode_{class_name}_{region}_finetune_BIG_100_ddim_l1_10.0-19.pt'
                else:
                    model_name = f'geode_{class_name}_{region}_finetune_NORMAL_30_ddim_l1_10.0-9.pt'
                    

                script = f'''! python main.py --generate_synth\
                --data_override GEODE\
                --config afhq.yml   \
                --exp ./runs/test        \
                --do_train 1             \
                --do_test 0 \
                --n_iter 1              \
                --t_0 500                \
                --n_inv_step 40     \
                --n_train_step 10 \
                --n_test_step 40 \
                --model_path checkpoint/{model_name} \
                --latent_file_path geode_{class_name}_{region}_finetune \
                --latent_mult 5 \
                --bandwidth {b}'''
            

                whole_slurm = boilerplate + "\n" + script
                b_str = str(b).replace('.',"d")
                file_name = f"slurm/geode/synth/synth_geode_{region}_{class_name}_{p_set}_b{b_str}.slurm"
                if True or os.path.exists(file_name):
                        with open(file_name, "w") as text_file:
                            print('creating',file_name)
                            text_file.write(whole_slurm)




