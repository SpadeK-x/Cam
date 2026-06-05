import argparse
import os

import lightning as pl
import numpy as np
import torch
from PIL import Image

from camclone_wan22_utils import (
    CamCloneDataset,
    count_parameters,
    freeze_except_lora,
    inject_lora_adapters,
    load_lora_state_dict,
    load_wan_i2v_pipeline,
    lora_state_dict,
)


class LightningModelForWan22CamClone(pl.LightningModule):
    def __init__(
        self,
        model_paths,
        learning_rate=5e-5,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        lora_rank=16,
        lora_alpha=16.0,
        lora_dropout=0.0,
        expert_scope="high",
        allow_no_expert_match=False,
        resume_lora_path=None,
        content_drop_prob=0.5,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
    ):
        super().__init__()
        self.pipe = load_wan_i2v_pipeline(model_paths, torch_dtype=torch.bfloat16, device="cpu")
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.pipe.requires_grad_(False)
        target_names = inject_lora_adapters(
            self.pipe.denoising_model(),
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            expert_scope=expert_scope,
            allow_no_expert_match=allow_no_expert_match,
        )
        freeze_except_lora(self.pipe.denoising_model())
        self.pipe.eval()
        self.pipe.denoising_model().train()

        if resume_lora_path is not None:
            load_lora_state_dict(self.pipe.denoising_model(), resume_lora_path)
            print(f"Loaded LoRA adapter from {resume_lora_path}")

        trainable, total = count_parameters(self.pipe.denoising_model())
        print(f"Injected {len(target_names)} LoRA layers:")
        for name in target_names:
            print(f"  LoRA: {name}")
        print(f"Trainable DiT parameters: {trainable} / {total}")

        self.learning_rate = learning_rate
        self.content_drop_prob = content_drop_prob
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    @staticmethod
    def _to_pil_list(first_frames):
        return [
            Image.fromarray(frame.detach().cpu().numpy().astype(np.uint8))
            if torch.is_tensor(frame)
            else Image.fromarray(frame.astype(np.uint8))
            for frame in first_frames
        ]

    def training_step(self, batch, batch_idx):
        text_s = batch["text"]
        video_s = batch["video"].to(dtype=self.pipe.torch_dtype, device=self.device)
        ref_video_s = batch["ref_video"].to(dtype=self.pipe.torch_dtype, device=self.device)
        content_video_s = batch["content_video"].to(dtype=self.pipe.torch_dtype, device=self.device)

        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text_s)
        latents = self.pipe.encode_video(video_s, **self.tiler_kwargs)
        ref_latents = self.pipe.encode_video(ref_video_s, **self.tiler_kwargs)
        content_latents = self.pipe.encode_video(content_video_s, **self.tiler_kwargs)

        if torch.rand((), device=self.device) < self.content_drop_prob:
            content_latents = torch.zeros_like(content_latents)

        _, _, num_frames, height, width = video_s.shape
        first_frame_s = self._to_pil_list(batch["first_frame"])
        image_emb = self.pipe.encode_image(first_frame_s, num_frames, height, width)

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.device)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        extra_input = self.pipe.prepare_extra_input(latents)

        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep=timestep,
            cam_emb=None,
            **prompt_emb,
            **extra_input,
            **image_emb,
            ref_latents=ref_latents,
            content_latents=content_latents,
            content_drop_prob=0.0,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )
        target_frames = latents.shape[2]
        loss = torch.nn.functional.mse_loss(
            noise_pred[:, :, :target_frames].float(),
            training_target[:, :, :target_frames].float(),
        )
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = (param for param in self.pipe.denoising_model().parameters() if param.requires_grad)
        return torch.optim.AdamW(params, lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint.clear()
        torch.save(
            lora_state_dict(self.pipe.denoising_model()),
            os.path.join(checkpoint_dir, f"camclone_wan22_lora_step{self.global_step}.pt"),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train CamCloneMaster on native Wan2.2 I2V")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, default="models/train_wan22")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_1")
    parser.add_argument("--tiled", default=False, action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--expert_scope", type=str, default="high", choices=["high", "high_low", "all"])
    parser.add_argument("--allow_no_expert_match", default=False, action="store_true")
    parser.add_argument("--resume_lora_path", type=str, default=None)
    parser.add_argument("--content_drop_prob", type=float, default=0.5)
    parser.add_argument("--no_gradient_checkpointing", dest="use_gradient_checkpointing", action="store_false")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true")
    parser.set_defaults(use_gradient_checkpointing=True)
    return parser.parse_args()


def train(args):
    dataset = CamCloneDataset(
        args.dataset_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )
    model = LightningModelForWan22CamClone(
        model_paths=args.model_paths,
        learning_rate=args.learning_rate,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        expert_scope=args.expert_scope,
        allow_no_expert_match=args.allow_no_expert_match,
        resume_lora_path=args.resume_lora_path,
        content_drop_prob=args.content_drop_prob,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=500, filename="{epoch}-{step}")],
        logger=None,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    train(parse_args())
