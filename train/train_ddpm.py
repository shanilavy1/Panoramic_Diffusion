from ddpm.diffusion import GaussianDiffusion, Trainer
import hydra
from omegaconf import DictConfig, open_dict
from train.get_dataset import get_dataset
import torch
import os
from ddpm.unet import UNet


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    # Create 2D UNet for X-ray diffusion with CT conditioning
    model = UNet(
        in_ch=cfg.model.diffusion_num_channels,   # 1 for grayscale X-ray
        out_ch=cfg.model.diffusion_num_channels,   # 1 for noise prediction
        spatial_dims=2,                             # 2D UNet for X-ray
    ).cuda()

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
        save_and_sample_every=cfg.model.save_and_sample_every,
        results_folder=cfg.model.results_folder,
        num_sample_rows=cfg.model.num_sample_rows,
        max_grad_norm=cfg.model.max_grad_norm,
        num_workers=cfg.model.num_workers,
        lora=cfg.model.lora,
        lora_first=cfg.model.lora_first,
        use_wandb=cfg.model.use_wandb,
        wandb_project=cfg.model.wandb_project,
        wandb_run_name=cfg.model.wandb_run_name,
    )

    # Load checkpoint if specified
    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone, map_location='cuda:0')

    # Start training
    trainer.train()


if __name__ == '__main__':
    run()
