from ddpm.diffusion import GaussianDiffusion, Trainer
import hydra
from omegaconf import DictConfig, open_dict
from train.get_dataset import get_dataset
import torch
import os
import random
import numpy as np
from ddpm.unet import UNet
from ddpm.unet_v2 import UNetV2


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    # Set seed FIRST, before any model/data creation
    set_seed(cfg.model.seed)

    torch.cuda.set_device(cfg.model.gpus)

    with open_dict(cfg):
        # Use wandb_run_name as subfolder so each experiment gets isolated results
        experiment_name = cfg.model.wandb_run_name
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, experiment_name)

    # Create 2D UNet for X-ray diffusion with CT conditioning
    unet_type = cfg.model.get('unet_type', 'v1')
    if unet_type == 'v2':
        model = UNetV2(
            in_ch=cfg.model.diffusion_num_channels,   # 1 for grayscale X-ray
            out_ch=cfg.model.diffusion_num_channels,   # 1 for noise prediction
            spatial_dims=2,                             # 2D UNet for X-ray
        ).cuda()
        print(f'Using UNetV2 (ResNet blocks + attention + FiLM CT conditioning)')
    else:
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,   # 1 for grayscale X-ray
            out_ch=cfg.model.diffusion_num_channels,   # 1 for noise prediction
            spatial_dims=2,                             # 2D UNet for X-ray
        ).cuda()
        print(f'Using UNet v1 (MONAI blocks + additive CT conditioning)')

    # Create diffusion model for 2D X-ray generation conditioned on CT
    diffusion = GaussianDiffusion(
        denoise_fn=model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        l1_weight=cfg.model.l1_weight,
        perceptual_weight=cfg.model.perceptual_weight,
        discriminator_weight=cfg.model.discriminator_weight,
        name_dataset=cfg.model.name_dataset,
    ).cuda()

    # Get datasets (split ratios come from config)
    train_dataset, val_dataset, test_dataset = get_dataset(cfg)

    # Create trainer with wandb integration and validation metrics
    trainer = Trainer(
        diffusion_model=diffusion,
        cfg=cfg,
        dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_batch_size=cfg.model.batch_size,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        step_start_ema=cfg.model.step_start_ema,
        update_ema_every=cfg.model.update_ema_every,
        validate_every_n_epochs=cfg.model.validate_every_n_epochs,
        results_folder=cfg.model.results_folder,
        num_sample_rows=cfg.model.num_sample_rows,
        max_grad_norm=cfg.model.max_grad_norm,
        num_workers=cfg.model.num_workers,
        lora=cfg.model.lora,
        lora_first=cfg.model.lora_first,
        use_wandb=cfg.model.use_wandb,
        wandb_project=cfg.model.wandb_project,
        wandb_run_name=cfg.model.wandb_run_name,
        seed=cfg.model.seed,
        weight_decay=cfg.model.get('weight_decay', 0.0),
        use_cosine_lr=cfg.model.get('use_cosine_lr', False),
    )

    # Load checkpoint if specified
    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone, map_location='cuda:0')

    # Start training
    trainer.train()


if __name__ == '__main__':
    run()
