import argparse
import os



script = f'''  ! python main.py --clip_finetune_eff        \
               --data_override GEODE\
               --config afhq.yml      \
               --exp ./runs/test        \
               --edit_attr house_in_Africa      \
               --do_train 1             \
               --do_test 0 \
               --n_train_img {30}         \
               --bs_train 2 \
               --n_iter 10               \
               --t_0 500                \
               --n_inv_step 40          \
               --lr_clip_finetune 8e-6  \
               --l1_loss_w 10            \
               --clip_loss_w 0 \
               --n_train_step 10 \
               --n_precomp_img 30 \
               --save_train_image 0 \
               --model_path pretrained/512x512_diffusion.pt \
               --model_save_name geode_house_Africa_finetune \
               --finetune_class_name house \
               --finetune_region Africa \
               --param_set NORMAL 

'''


parser = argparse.ArgumentParser(description=globals()["__doc__"])


parser.add_argument('--action', type=str, default="finetune")
parser.add_argument('--dataset', type=str, default='geode')
parser.add_argument('--big', type=int, default=0)


classes = ['house', 'spices', 'religious_building', 'hand_soap', 'dustbin', "medicine",
           "car", 
           "storefront", 
           "plate_of_food",
            ]
regions = ["Africa", "Americas", "EastAsia", "Europe", "SouthEastAsia", "WestAsia"]



args = parser.parse_args()

if args.action == 'finetune':
    if args.dataset == 'geode':

        param_set = "normal"
        if args.big == 1:
            n_img = 100
            n_iter = 20
            param_set= "big"
            time_str="08:00:00"
        else:
            n_img = 30
            n_iter = 10
            time_str="01:00:00"


        for class_name in classes:
            for region in regions:


                boilerplate = f'''#!/bin/bash
#SBATCH --job-name=geode_finetune_{region}_{class_name}_{param_set}    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time={time_str}          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=akelch@princeton.edu
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu80
#SBATCH --mem=5G
module purge
module load anaconda3/2023.9
conda activate torch-thesis

'''

                script = \
                    f'''  ! python main.py --clip_finetune_eff        \
                    --data_override GEODE\
                    --config afhq.yml      \
                    --exp ./runs/test        \
                    --do_train 1             \
                    --do_test 0 \
                    --n_train_img {n_img}         \
                    --bs_train 2 \
                    --n_iter {n_iter}               \
                    --t_0 500                \
                    --n_inv_step 40          \
                    --lr_clip_finetune 8e-6  \
                    --l1_loss_w 10            \
                    --clip_loss_w 0 \
                    --n_train_step 10 \
                    --n_precomp_img {n_img} \
                    --save_train_image 0 \
                    --model_path pretrained/512x512_diffusion.pt \
                    --model_save_name geode_{class_name}_{region}_finetune \
                    --finetune_class_name {class_name} \
                    --finetune_region {region} \
                    --param_set {"NORMAL" if args.big == 0 else "BIG"} 

                    '''
                
                whole_slurm = "\n".join([boilerplate, script])

                # save file
                file_name = f"slurm/{args.dataset}/{args.action}/{args.dataset}_{args.action}_{region}_{class_name}_{param_set}.slurm"
                if True or os.path.exists(file_name):
                    with open(file_name, "w") as text_file:
                        print('creating', f"slurm/{args.dataset}/{args.action}/{args.dataset}_{args.action}_{region}_{class_name}.slurm")
                        text_file.write(whole_slurm)
                
                




