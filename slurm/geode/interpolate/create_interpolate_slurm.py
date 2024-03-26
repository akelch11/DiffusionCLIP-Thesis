
import argparse
import os



parser = argparse.ArgumentParser(description=globals()["__doc__"])
parser.add_argument("--mult", type=int, default=5)
parser.add_argument("--lambda_step", type=float, default=0.25)

args = parser.parse_args()

# p_set = "big" if args.big == 1 else "normal"
# time_str = "01:00:00" if args.big == 0 else "13:00:00"

classes = [
            #     'house', 
            #    'spices', 
            #    'religious_building', 
            #    'hand_soap',
            #     'dustbin',
            #     "medicine"
                # "car", 
                # "storefront",
               "plate_of_food"
                ]
regions = ["Africa", "Americas", "EastAsia", "Europe", "SouthEastAsia", "WestAsia"]


for class_name in classes:
    for region in regions:
        for p_set in ['NORMAL', "BIG"]:






            boilerplate = f'''#!/bin/bash
#SBATCH --job-name=geode_interpolate_latents_{region}_{class_name}  # create a short name for your job
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
#SBATCH --mem=40G

module purge
module load anaconda3/2023.9
conda activate torch-thesis
'''

            if p_set == "NORMAL":
                model_name = f'geode_{class_name}_{region}_finetune_{p_set}_30_ddim_l1_10.0-9.pt'
            else:
                model_name = f'geode_{class_name}_{region}_finetune_{p_set}_100_ddim_l1_10.0-19.pt'

            
            script = f''' ! python main.py --interpolate_latents \
                    --data_override GEODE \
                    --config afhq.yml \
                    --exp ./runs/test  \
                    --do_train 1 \
                    --do_test 0 \
                    --n_iter 1  \
                    --t_0 500  \
                    --n_inv_step 40   \
                    --n_train_step 10 \
                    --n_test_step 40 \
                    --model_path checkpoint/{model_name} \
                    --model_save_name geode_{class_name}_{region}_finetune \
                    --finetune_class_name {class_name} \
                    --finetune_region {region} \
                    --latent_mult {args.mult}'''
            
            whole_slurm = boilerplate + "\n" + script
            file_name = f"slurm/geode/interpolate/interpolate_geode_{region}_{class_name}_{p_set}.slurm"
            if True or os.path.exists(file_name):
                    with open(file_name, "w") as text_file:
                        print('creating',file_name)
                        text_file.write(whole_slurm)
