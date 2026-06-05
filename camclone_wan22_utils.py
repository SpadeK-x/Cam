import glob
import json
import os
from typing import Iterable, List, Optional, Tuple

import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffsynth import ModelManager, WanVideoI2VPipeline
from diffsynth.models.utils import init_weights_on_device, load_state_dict
from diffsynth.models.wan_video_dit import WanModel
from einops import rearrange
from PIL import Image
from torchvision.transforms import v2


class CamCloneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        max_num_frames=81,
        frame_interval=1,
        num_frames=81,
        height=480,
        width=832,
        infer=False,
    ):
        metadata = pd.read_csv(csv_path)
        self.text = metadata["caption"].to_list()
        self.ref_path = metadata["ref_video_path"].to_list()
        self.path = metadata["video_path"].to_list() if "video_path" in metadata.columns else None
        self.content_path = metadata["content_video_path"].to_list() if "content_video_path" in metadata.columns else None
        self.first_frame_path = metadata["first_frame_path"].to_list() if "first_frame_path" in metadata.columns else None
        if not infer and (self.path is None or self.content_path is None):
            raise ValueError("Training CSV must contain video_path and content_video_path columns.")
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.infer = infer
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.image_process = v2.Compose([v2.CenterCrop(size=(height, width))])

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        return torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )

    def load_frames_using_imageio(self, file_path):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < self.max_num_frames or reader.count_frames() - 1 < (self.num_frames - 1) * self.frame_interval:
            frame_indices = [min(frame_index, reader.count_frames() - 1) for frame_index in range(self.num_frames)]
        else:
            frame_indices = list(range(self.num_frames))

        frames = []
        first_frame = None
        for frame_id in frame_indices:
            frame = Image.fromarray(reader.get_data(frame_id * self.frame_interval))
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(self.image_process(frame))
            frames.append(self.frame_process(frame))
        reader.close()
        frames = torch.stack(frames, dim=0)
        return rearrange(frames, "T C H W -> C T H W"), first_frame

    def load_video(self, file_path):
        return self.load_frames_using_imageio(file_path)

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        return np.array(self.image_process(frame))

    def __getitem__(self, data_id):
        while True:
            try:
                text = self.text[data_id]
                ref_video, _ = self.load_video(self.ref_path[data_id])
                first_frame = None
                content_video = None
                content_path = None

                if self.content_path is not None:
                    content_path = self.content_path[data_id]
                    content_video, first_frame = self.load_video(content_path)
                if self.first_frame_path is not None:
                    first_frame = self.load_image(self.first_frame_path[data_id])

                data = {
                    "text": text,
                    "first_frame": first_frame,
                    "ref_video": ref_video,
                    "ref_path": self.ref_path[data_id],
                    "content_video": content_video,
                    "content_path": content_path,
                }
                if not self.infer:
                    video, video_first_frame = self.load_video(self.path[data_id])
                    data["video"] = video
                    data["path"] = self.path[data_id]
                    if first_frame is None:
                        data["first_frame"] = video_first_frame
                return data
            except Exception as exc:
                if self.infer:
                    raise exc
                data_id = (data_id + 1) % len(self.text)

    def __len__(self):
        return len(self.text)


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank=16, alpha=16.0, dropout=0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


ATTENTION_KEYWORDS = ("self_attn", "self_attention", "attn1", "temporal_attn", "spatial_temporal_attn")
PROJECTION_NAMES = ("q", "k", "v", "o", "to_q", "to_k", "to_v", "to_out", "out_proj")
HIGH_NOISE_KEYWORDS = ("high", "high_noise", "highnoise")
LOW_NOISE_KEYWORDS = ("low", "low_noise", "lownoise")


def _is_checkpoint_file(path):
    return os.path.isfile(path) and path.split(".")[-1] in {"safetensors", "bin", "ckpt", "pth", "pt"}


def _checkpoint_files_in_dir(path):
    files = []
    for pattern in ("*.safetensors", "*.bin", "*.ckpt", "*.pth", "*.pt"):
        files.extend(glob.glob(os.path.join(path, pattern)))
    return sorted(files)


def normalize_model_path_groups(model_paths: Iterable[str]):
    expanded_paths = []
    for raw_path in model_paths:
        if not raw_path:
            continue
        matches = sorted(glob.glob(raw_path)) if glob.has_magic(raw_path) else [raw_path]
        expanded_paths.extend(matches)

    if not expanded_paths:
        raise ValueError("At least one Wan2.2 I2V model path must be provided via --model_paths.")

    groups = []
    shard_files_by_parent = {}
    for path in expanded_paths:
        if os.path.isdir(path):
            checkpoint_files = _checkpoint_files_in_dir(path)
            if checkpoint_files:
                groups.append(checkpoint_files)
                continue
            if os.path.exists(os.path.join(path, "config.json")):
                groups.append(path)
                continue
            raise ValueError(f"Model directory contains no supported checkpoint files: {path}")
        if _is_checkpoint_file(path):
            parent = os.path.dirname(path)
            sibling_shards = _checkpoint_files_in_dir(parent)
            if len(sibling_shards) > 1 and path in sibling_shards:
                shard_files_by_parent[parent] = sibling_shards
            else:
                groups.append(path)
            continue
        if os.path.exists(path):
            groups.append(path)
            continue
        raise FileNotFoundError(f"Model path does not exist: {path}")

    seen_parent_groups = set()
    for parent, shard_files in shard_files_by_parent.items():
        key = tuple(shard_files)
        if key not in seen_parent_groups:
            groups.append(shard_files)
            seen_parent_groups.add(key)
    return groups


def _as_wan_model_config_path(model_path):
    if isinstance(model_path, list):
        if not model_path:
            return None
        parent = os.path.dirname(model_path[0])
    elif os.path.isdir(model_path):
        parent = model_path
    else:
        return None
    config_path = os.path.join(parent, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        config = json.load(f)
    architectures = config.get("architectures") or [config.get("_class_name")]
    if "WanModel" not in architectures:
        return None
    return config_path


def _wan_model_kwargs_from_config(config):
    return {
        "has_image_input": config.get("has_image_input", config.get("in_dim") == 36),
        "patch_size": tuple(config.get("patch_size", (1, 2, 2))),
        "in_dim": config["in_dim"],
        "dim": config["dim"],
        "ffn_dim": config["ffn_dim"],
        "freq_dim": config.get("freq_dim", 256),
        "text_dim": config.get("text_dim", 4096),
        "out_dim": config.get("out_dim", 16),
        "num_heads": config["num_heads"],
        "num_layers": config["num_layers"],
        "eps": config.get("eps", 1e-6),
    }


def _strip_state_dict_prefix(state_dict, prefix):
    if not state_dict or not all(key.startswith(prefix) for key in state_dict):
        return None
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def _wan_state_dict_candidates(state_dict):
    candidates = [("original", state_dict)]
    for prefix in ("model.", "module.", "dit.", "transformer."):
        stripped = _strip_state_dict_prefix(state_dict, prefix)
        if stripped is not None:
            candidates.append((f"strip:{prefix}", stripped))
    for label, candidate in list(candidates):
        converted, _ = WanModel.state_dict_converter().from_diffusers(candidate)
        if converted:
            candidates.append((f"{label}:from_diffusers", converted))
    return candidates


def _load_wan_model_from_config_shards(model_manager, model_path, torch_dtype, device):
    config_path = _as_wan_model_config_path(model_path)
    if config_path is None:
        return False
    with open(config_path, "r") as f:
        config = json.load(f)

    shard_paths = model_path if isinstance(model_path, list) else _checkpoint_files_in_dir(model_path)
    state_dict = {}
    for shard_path in shard_paths:
        state_dict.update(load_state_dict(shard_path, torch_dtype=torch_dtype))

    load_errors = []
    for label, candidate_state_dict in _wan_state_dict_candidates(state_dict):
        with init_weights_on_device():
            model = WanModel(**_wan_model_kwargs_from_config(config))
        try:
            missing_keys, unexpected_keys = model.load_state_dict(candidate_state_dict, strict=False, assign=True)
        except RuntimeError as exc:
            load_errors.append((label, str(exc).splitlines()[:10]))
            continue
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            break
        load_errors.append((label, [f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}", f"missing[:10]={missing_keys[:10]}", f"unexpected[:10]={unexpected_keys[:10]}"]))
    else:
        print("    WanModel config-based loading failed for all key candidates:")
        for label, lines in load_errors:
            print(f"    Candidate {label}:")
            for line in lines:
                print(f"      {line}")
        raise RuntimeError(
            "Wan2.2 WanModel checkpoint did not match the local WanModel architecture. "
            "The loader reached the config-based branch, but key names/shapes still need a converter."
        )
    model = model.eval().to(dtype=torch_dtype, device=device)
    model_manager.model.append(model)
    model_manager.model_path.append(os.path.dirname(config_path))
    model_manager.model_name.append("wan_video_dit")
    print(f"    Loaded WanModel from sharded config folder: {os.path.dirname(config_path)}")
    return True


def load_wan_i2v_pipeline(model_paths: Iterable[str], torch_dtype=torch.bfloat16, device="cpu"):
    model_paths = normalize_model_path_groups(model_paths)
    if not model_paths:
        raise ValueError("At least one Wan2.2 I2V model path must be provided via --model_paths.")
    model_manager = ModelManager(torch_dtype=torch_dtype, device=device)
    for model_path in model_paths:
        if not _load_wan_model_from_config_shards(model_manager, model_path, torch_dtype, device):
            model_manager.load_model(model_path)
    pipe = WanVideoI2VPipeline.from_model_manager(model_manager)
    missing = [
        name
        for name in ("text_encoder", "image_encoder", "vae", "dit")
        if getattr(pipe, name, None) is None
    ]
    if missing:
        raise RuntimeError(
            "Failed to load a complete native Wan I2V pipeline. Missing: "
            f"{', '.join(missing)}. This local DiffSynth build may not recognize Wan2.2 checkpoints yet; "
            "add/update Wan2.2 model loader support, then rerun this script."
        )
    return pipe


def pad_latents_to_channels(latents, target_channels):
    if latents.shape[1] == target_channels:
        return latents
    pad_channels = target_channels - latents.shape[1]
    if pad_channels < 0:
        raise ValueError(f"Latent channels ({latents.shape[1]}) exceed target channels ({target_channels}).")
    return F.pad(latents, (0, 0, 0, 0, 0, 0, 0, pad_channels))


def _matches_expert_scope(name: str, expert_scope: str):
    lowered = name.lower()
    has_high = any(keyword in lowered for keyword in HIGH_NOISE_KEYWORDS)
    has_low = any(keyword in lowered for keyword in LOW_NOISE_KEYWORDS)
    if expert_scope == "all":
        return True
    if expert_scope == "high":
        return has_high
    if expert_scope == "high_low":
        return has_high or has_low
    raise ValueError(f"Unknown expert_scope: {expert_scope}")


def _matches_lora_target(name: str, module: nn.Module, expert_scope: str):
    if not isinstance(module, nn.Linear):
        return False
    lowered = name.lower()
    if not any(keyword in lowered for keyword in ATTENTION_KEYWORDS):
        return False
    if name.split(".")[-1] not in PROJECTION_NAMES:
        return False
    return _matches_expert_scope(name, expert_scope)


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module):
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora_adapters(
    model: nn.Module,
    rank=16,
    alpha=16.0,
    dropout=0.0,
    expert_scope="high",
    allow_no_expert_match=False,
):
    targets: List[Tuple[str, nn.Linear]] = [
        (name, module)
        for name, module in model.named_modules()
        if _matches_lora_target(name, module, expert_scope)
    ]
    if not targets and not allow_no_expert_match:
        raise RuntimeError(
            "No LoRA target modules matched. For Wan2.2 MoE, check high-noise expert module names; "
            "for a smoke test on a non-MoE Wan model, pass --expert_scope all."
        )
    for name, module in targets:
        _set_module(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
    return [name for name, _ in targets]


def freeze_except_lora(model: nn.Module):
    for _, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad = True


def lora_state_dict(model: nn.Module):
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }


def load_lora_state_dict(model: nn.Module, path: str):
    state_dict = torch.load(path, map_location="cpu")
    return model.load_state_dict(state_dict, strict=False)


def count_parameters(model: nn.Module):
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total
