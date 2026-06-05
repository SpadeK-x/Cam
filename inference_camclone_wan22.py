import argparse
import os

import torch
from diffsynth import save_video

from camclone_wan22_utils import (
    CamCloneDataset,
    inject_lora_adapters,
    load_lora_state_dict,
    load_wan_i2v_pipeline,
)


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(description="CamCloneMaster Wan2.2 inference")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="demo/camclone_wan22_output")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cameraclone_type", type=str, default="i2v", choices=["i2v", "v2v"])
    parser.add_argument("--no_tiled", dest="tiled", action="store_false")
    parser.add_argument("--tile_size_height", type=int, default=30)
    parser.add_argument("--tile_size_width", type=int, default=52)
    parser.add_argument("--tile_stride_height", type=int, default=15)
    parser.add_argument("--tile_stride_width", type=int, default=26)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--expert_scope", type=str, default="high", choices=["high", "high_low", "all"])
    parser.add_argument("--allow_no_expert_match", default=False, action="store_true")
    parser.set_defaults(tiled=True)
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = load_wan_i2v_pipeline(args.model_paths, torch_dtype=torch.bfloat16, device="cpu")
    inject_lora_adapters(
        pipe.denoising_model(),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        expert_scope=args.expert_scope,
        allow_no_expert_match=args.allow_no_expert_match,
    )
    load_lora_state_dict(pipe.denoising_model(), args.lora_path)
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    dataset = CamCloneDataset(
        args.dataset_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        infer=True,
    )

    for data_id in range(len(dataset)):
        item = dataset[data_id]
        if item["first_frame"] is None:
            raise ValueError("Wan2.2 I2V inference requires first_frame_path or content_video_path.")
        if args.cameraclone_type == "v2v" and item["content_video"] is None:
            raise ValueError("V2V camera clone requires content_video_path.")

        input_image = torch.from_numpy(item["first_frame"]).unsqueeze(0)
        ref_video = item["ref_video"].unsqueeze(0)
        content_video = item["content_video"].unsqueeze(0) if item["content_video"] is not None else None

        video = pipe(
            prompt=[item["text"]],
            negative_prompt=NEGATIVE_PROMPT,
            input_image=input_image,
            ref_video=ref_video,
            content_video=content_video,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            tiled=args.tiled,
            tile_size=(args.tile_size_height, args.tile_size_width),
            tile_stride=(args.tile_stride_height, args.tile_stride_width),
            cameraclone_type=args.cameraclone_type,
        )
        output_path = os.path.join(args.output_dir, f"camclone_wan22_{data_id:03d}.mp4")
        save_video(video, output_path, fps=30, quality=5)


if __name__ == "__main__":
    main(parse_args())
