import time
# copyy
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
import pickle
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
from matplotlib import pyplot as plt

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment

class DiffusionCLIP(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]

        self.finetune_class_name = args.finetune_class_name
        self.finetune_region = args.finetune_region


    # def generate_images_from_model_and_data(self,model_path, dataset, class_name=None, region=None):
    

    def interpolate_latents_from_dataset(self, M=1):

        print(self.args.exp)
        models = []
        model_paths = [None, self.args.model_path]
        for model_path in model_paths:
            if self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
                model_i = i_DDPM(self.config.data.dataset)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
                learn_sigma = True
            else:
                print('Not implemented dataset')
                raise ValueError
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        # ----------- Precompute Latents thorugh Inversion Process -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        n = 1
        img_lat_pairs_dic = {}
        for mode in ['train']:
            img_lat_pairs = []
            # pairs_path = os.path.join('precomputed/',
            #                           f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            pairs_path = ''
            # if os.path.exists(pairs_path):
            #     print(f'{mode} pairs exists')
            #     img_lat_pairs_dic[mode] = torch.load(pairs_path)
            #     for step, (x0, x_id, e_id) in enumerate(img_lat_pairs_dic[mode]):
            #         tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
            #         tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_1_rec.png'))
            #         if step == self.args.n_precomp_img - 1:
            #             break
            #     continue
            # else:
            train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
            loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
            loader = loader_dic[mode]

            
            L_STEP = 0.3

            for step, img in enumerate(loader):
                img_1_latent = self.invert_image(img)
                for m in range(M): # multiplicty of latents
                    rng_index = np.random.randint(0, len(loader))
                    img_2_latent = loader[rng_index].clone()

                    for lambda_step in [L_STEP, 1 - L_STEP]
                        new_latent = img_1_latent + lambda_step * (img_2_latent - img_1_latent)
                    
                    

                # x0 = img.to(self.config.device)
                # tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                # x = x0.clone()
                # with torch.no_grad():
                #     with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                #         for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                #             t = (torch.ones(n) * i).to(self.device)
                #             t_prev = (torch.ones(n) * j).to(self.device)

                #             x = denoising_step(x, t=t, t_next=t_prev, models=models,
                #                                logvars=self.logvar,
                #                                sampling_type='ddim',
                #                                b=self.betas,
                #                                eta=0,
                #                                learn_sigma=learn_sigma,
                #                                ratio=0)

                #             progress_bar.update(1)

                #     x_lat = x.clone()
                #     tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                #                                                    f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    # with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                    #     for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                    #         t = (torch.ones(n) * i).to(self.device)
                    #         t_next = (torch.ones(n) * j).to(self.device)

                    #         x = denoising_step(x, t=t, t_next=t_next, models=models,
                    #                            logvars=self.logvar,
                    #                            sampling_type=self.args.sample_type,
                    #                            b=self.betas,
                    #                            eta=self.args.eta,
                    #                            learn_sigma=learn_sigma,
                    #                            ratio=0)

                    #         progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])

                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_1_rec.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)
    
    def clip_finetune_eff(self):
        print(self.args.exp)
        print(f'   {self.src_txts}')
        print(f'-> {self.trg_txts}')

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            print('FORCING IMAGNET DDPM CREATION')
            # model = i_DDPM(self.config.data.dataset)
            model = i_DDPM("IMAGENET")
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        # optim_ft = torch.optim.SGD(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)#, momentum=0.9)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)
        
        id_loss_func = id_loss.IDLoss().to(self.device).eval()
        

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}
        modes = ['train', 'test']
        for mode in ['train']:
            img_lat_pairs = []
            if self.args.edit_attr in ['female', 'male']:
                self.config.data.dataset = 'GENDER'
                self.config.data.category = 'GENDER'
                if self.args.edit_attr == 'female':
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_male_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_female_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            elif self.config.data.dataset == "IMAGENET":
                if self.args.target_class_num is not None:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{IMAGENET_DIC[str(self.args.target_class_num)][1]}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.args.data_override}_{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            else:
                # pairs_path = os.path.join('precomputed/',
                #                           f'{self.args.data_override}_{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                pairs_path = os.path.join('precomputed/',f'{self.args.model_save_name}_{self.args.param_set}')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path, map_location=torch.device('cpu'))
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                if self.args.edit_attr == 'female':
                    train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              gender='male')
                elif self.args.edit_attr == 'male':
                    train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              gender='female')
                elif self.args.data_override:
                    print(f'FORCING TO USE DATASET {self.args.data_override} for FINETUNE')
                    if self.args.data_override == 'GEODE':
                        train_dataset, test_dataset = get_dataset(self.args.data_override, DATASET_PATHS, self.config,
                                                              target_class_num=self.args.target_class_num, \
                                                              class_name=self.finetune_class_name, region=self.finetune_region)
                    else:
                        train_dataset, test_dataset = get_dataset(self.args.data_override, DATASET_PATHS, self.config,
                                                              target_class_num=self.args.target_class_num, class_name=self.finetune_class_name)
                else:
                    print('default to AFHQ')
                    train_dataset, test_dataset = get_dataset('AFHQ', DATASET_PATHS, self.config,
                                                              target_class_num=self.args.target_class_num, class_name=self.finetune_class_name)
                    

                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]
            print(len(loader), 'images in loader')
            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                # tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                print('image shape', x0.shape)
                x = x0.clone()
                model.eval()
                time_s = time.time()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    time_e = time.time()
                    print(f'{time_e - time_s} seconds')
                    x_lat = x.clone()
                    print('x lat shape', x_lat.shape)
                    # tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                    #                                               f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        time_s = time.time()
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)
                        time_e = time.time()
                        print(f'{time_e - time_s} seconds')

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            # pairs_path = os.path.join('precomputed/',
            #                           f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            model.module.load_state_dict(init_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_loss_func.target_direction = None
            iter_losses = []
            # ----------- Train -----------#
            for it_out in range(self.args.n_iter):

                iter_loss = 0
                last_loss = 0

                exp_id = os.path.split(self.args.exp)[-1]
                print('save name parts', exp_id, trg_txt, it_out)
                save_name = f'checkpoint/{exp_id}_{trg_txt.replace(" ", "_")}-{it_out}.pth'
                if self.args.model_save_name:
                    # save_name = f'checkpoint/{self.args.model_save_name}-{it_out}.pth'
                    # full_model_save_name = f'checkpoint/{self.args.model_save_name}-{it_out}.pt'
                    save_name = f'checkpoint/{self.args.model_save_name}_{self.args.param_set}.pth'
                    full_model_save_name = f'checkpoint/{self.args.model_save_name}_{self.args.param_set}.pt'
                if self.args.do_train:
                    if os.path.exists(save_name):
                        print(f'{save_name} already exists.')
                        model.module.load_state_dict(torch.load(save_name))
                        continue
                    else:
                        for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic['train']):
                            model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone().to(self.device)
                            x0 = x0.to(self.device)
                            with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                                for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x, x0_t = denoising_step(x, t=t, t_next=t_next, models=model,
                                                             logvars=self.logvar,
                                                             sampling_type=self.args.sample_type,
                                                             b=self.betas,
                                                             eta=self.args.eta,
                                                             learn_sigma=learn_sigma,
                                                             out_x0_t=True)

                                    progress_bar.update(1)
                                    x = x.detach().clone()

                                    loss_clip = -torch.log((2 - clip_loss_func(x0, src_txt, x0_t, trg_txt)) / 2)
                                    loss_l1 = nn.L1Loss()(x0, x0_t)
                                    loss = self.args.clip_loss_w * loss_clip + self.args.l1_loss_w * loss_l1
                                    # if self.args.id_loss_w != 0:
                                    #     print('using ID loss')
                                    #     loss_id = torch.mean(id_loss_func(x0, x))
                                    #     loss += self.args.id_loss_w * loss_id
                                    loss.backward()

                                    
                                    iter_loss += loss
                                    if it_out == self.args.n_iter - 1:
                                        last_loss += loss

                                    optim_ft.step()
                                    for p in model.module.parameters():
                                        p.grad = None
                                    print(f"CLIP {step}-{it_out}: loss_clip: {loss_clip:.3f}")
                                    # break

                            if self.args.save_train_image:
                                extra = "ID" if self.args.id_loss_w > 0 else f"{trg_txt.replace(" ", "_")}{trg_txt.replace(" ", "_")}"
                                save_train_name = f'{self.args.data_override}_{extra})train_{step}_{it_out}_ngen_{self.args.n_train_step}.png'
                                tvu.save_image((x0_t + 1) * 0.5, os.path.join(self.args.image_folder, save_train_name))
                            time_in_end = time.time()
                            print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
                            if step == self.args.n_train_img - 1:
                                break

                        # Tracking Loss for Plot
                        iter_losses.append(iter_loss.item())
                        print('appending to loss,', iter_loss.item())
                        if it_out == self.args.n_iter-1:
                            iter_losses.append(last_loss.item())
                        

                        if it_out == self.args.n_iter-1:
                            if isinstance(model, nn.DataParallel):
                                torch.save(model.module.state_dict(), save_name)
                            else:
                                torch.save(model.state_dict(), save_name)
                            torch.save(model, full_model_save_name) # same complete model obj for loading later
                            print(f'Model {save_name} is saved.')



                        scheduler_ft.step()

                # ----------- Eval -----------#
                if self.args.do_test:
                    if not self.args.do_train:
                        print(save_name)
                        model.module.load_state_dict(torch.load(save_name))

                    model.eval()
                    img_lat_pairs = img_lat_pairs_dic[mode]
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        with torch.no_grad():
                            x = x_lat.clone().to(self.device)
                            x0 = x0.to(self.device)
                            with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)

                            print(f"Eval {step}-{it_out}")
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'{mode}_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png'))
                            if step == self.args.n_test_img - 1:
                                break

            pickle.dump(iter_losses,f'plots/losses_{f'{self.args.data_override}_{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth'}')
            # iter_values = np.arange(0, len(iter_losses))
            # plt.plot(iter_values, iter_losses)
            # plt.title('Loss vs Fine-Tuning Iterations')
            # plt.xlabel("Fine Tuning Iterations")
            # plt.ylabel("Loss")
            # plt.savefig(f'plots/plot_{self.args.save_name}.png')

    def invert_image(self, img, ret_x0=False):
                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=models,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma,
                                               ratio=0)

                            progress_bar.update(1)

                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))
                    progress_bar.update(1)

                    # img_lat_pairs.append([x0, x_lat.detach().clone()])
                    if ret_x0:
                        return [x0,x_lat]
                    else:
                        return x_lat

                # tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_1_rec.png'))
                # if step == self.args.n_precomp_img - 1:
                #     break



    def edit_images_from_dataset(self):
        # ----------- Models -----------#
        print(self.args.exp)
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        models = []
        model_paths = [None, self.args.model_path]
        for model_path in model_paths:
            if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
                model_i = DDPM(self.config)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                learn_sigma = False
            elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
                model_i = i_DDPM(self.config.data.dataset)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
                learn_sigma = True
            else:
                print('Not implemented dataset')
                raise ValueError
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        # ----------- Precompute Latents thorugh Inversion Process -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        n = 1
        img_lat_pairs_dic = {}
        for mode in ['test']:
            img_lat_pairs = []
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, e_id) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_1_rec.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, img in enumerate(loader): # for each image, sample a random pair
                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=models,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma,
                                               ratio=0)

                            progress_bar.update(1)

                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=models,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               eta=self.args.eta,
                                               learn_sigma=learn_sigma,
                                               ratio=0)

                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])

                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_1_rec.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)


        # ----------- Generative Process -----------#
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_test_step != 0:
            seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
            seq_test = [int(s) for s in list(seq_test)]
            print('Uniform skip type')
        else:
            seq_test = list(range(self.args.t_0))
            print('No skip')
        seq_test_next = [-1] + list(seq_test[:-1])
        print("Start evaluation")
        eval_modes = ['test']
        for mode in eval_modes:

            img_lat_pairs = img_lat_pairs_dic[mode]
            for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):

                with torch.no_grad():
                    x = x_lat
                    with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                        for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=models,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               eta=self.args.eta,
                                               learn_sigma=learn_sigma,
                                               ratio=self.args.model_ratio,
                                               hybrid=self.args.hybrid_noise,
                                               hybrid_config=HYBRID_CONFIG)

                            progress_bar.update(1)

                    print(f"Eval {step}")
                    tvu.save_image((x + 1) * 0.5,
                                   os.path.join(self.args.image_folder,
                                                f'{mode}_{step}_2_clip_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))


    def edit_one_image(self):
        # ----------- Data -----------#
        n = self.args.bs_test

        if self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"]:
            try:
                img = run_alignment(self.args.img_path, output_size=self.config.data.image_size)
            except:
                img = Image.open(self.args.img_path).convert("RGB")
        else:
            img = Image.open(self.args.img_path).convert("RGB")
        img = img.resize((self.config.data.image_size, self.config.data.image_size), Image.ANTIALIAS)
        img = np.array(img)/255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f'0_orig.png'))
        x0 = (img - 0.5) * 2.

        # ----------- Models -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        models = []

        if self.args.hybrid_noise:
            model_paths = [None] + HYBRID_MODEL_PATHS
        else:
            model_paths = [self.args.model_path]

        # for model_path in model_paths:
        #     print('attempting load', model_path)
        #     if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
        #         model_i = DDPM(self.config)
        #         if model_path:
        #             ckpt = torch.load(model_path)
        #         else:
        #             ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
        #         learn_sigma = False
        #     elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
        #         model_i = i_DDPM(self.config.data.dataset)
        #         if model_path:
        #             ckpt = torch.load(model_path)
        #             # print('loading', ckpt)
        #         else:
        #             ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
        #             print('loading',MODEL_PATHS[self.config.data.dataset])
        #         learn_sigma = True
        #     else:
        #         print('Not implemented dataset')
        #         raise ValueError
                
        
        
        
        #     model_i.load_state_dict(ckpt)
        #     model_i.to(self.device)
        #     model_i = torch.nn.DataParallel(model_i)
        #     model_i.eval()
        #     print(f"{model_path} is loaded.")
        #     models.append(model_i)
        model_i = torch.load('checkpoint/best_model_ever_imagenet_finetune.pt')
        model_i.to(self.device)
        model_i = torch.nn.DataParallel(model_i)
        model_i.eval()
        
        models = model_i
        print('solo model')
        learn_sigma=True

        with torch.no_grad():
            #---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
            if self.args.deterministic_inv:
                x_lat_path = os.path.join(self.args.image_folder, f'x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
                if not os.path.exists(x_lat_path):
                    seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                    seq_inv = [int(s) for s in list(seq_inv)]
                    seq_inv_next = [-1] + list(seq_inv[:-1])

                    x = x0.clone()
                    with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=models,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma,
                                               ratio=0,
                                               )

                            progress_bar.update(1)
                        x_lat = x.clone()
                        torch.save(x_lat, x_lat_path)
                else:
                    print('Latent exists.')
                    x_lat = torch.load(x_lat_path)


            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_test_step}/{self.args.t_0}")
            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print('Uniform skip type')
            else:
                seq_test = list(range(self.args.t_0))
                print('No skip')
            seq_test_next = [-1] + list(seq_test[:-1])

            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[self.args.t_0 - 1].sqrt() + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'1_lat_ninv{self.args.n_inv_step}.png'))

                with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)

                        x = denoising_step(x, t=t, t_next=t_next, models=models,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           eta=self.args.eta,
                                           learn_sigma=learn_sigma,
                                           ratio=self.args.model_ratio,
                                           hybrid=self.args.hybrid_noise,
                                           hybrid_config=HYBRID_CONFIG)

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png'))
                        progress_bar.update(1)

                x0 = x.clone()
                if self.args.model_path:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}_{self.args.model_path.split('/')[-1].replace('.pth','')}.png"))
                else:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))

    