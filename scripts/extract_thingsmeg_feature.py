import os, sys
import torch
import numpy as np
import libs.autoencoder
import libs.clip
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from termcolor import cprint

import clip

from datasets import ThingsMEGDatabase

from nd.models.brain_encoder import BrainEncoder
from nd.utils.layout import ch_locations_2d
from nd.utils.eval_utils import get_run_dir


@torch.no_grad()
@hydra.main(
    version_base=None, config_path="../../configs/thingsmeg", config_name="clip"
)
def main(args):
    device = "cuda"

    # -----------------
    #      Dataset
    # -----------------
    dataset = ThingsMEGDatabase(args)
    train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
    test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)

    # -----------------
    #       Models
    # -----------------
    # Stable Diffusion
    autoencoder = libs.autoencoder.get_model(
        "assets/stable-diffusion/autoencoder_kl.pth"
    ).to(device)

    # CLIP-Vision ViT-B/32
    # clip_model, preprocess = clip.load("ViT-B/32")
    # clip_model = clip_model.eval().to(device)

    # CLIP-MEG
    subjects = dataset.subject_names if hasattr(dataset, "subject_names") else dataset.num_subjects  # fmt: skip
    brain_encoder = BrainEncoder(
        args,
        subjects=subjects,
        layout=eval(args.layout),
        vq=args.vq,
        blocks=args.blocks,
        downsample=args.downsample,
        temporal_aggregation=args.temporal_aggregation,
    ).to(device)
    brain_encoder.load_state_dict(
        torch.load(
            os.path.join(args.root, get_run_dir(args), "brain_encoder_best.pt"),
            map_location=device,
        )
    )
    brain_encoder.eval()

    # -----------------
    #      Extract
    # -----------------
    save_root = os.path.join(args.root, "data/uvit/thingsmeg_features")

    # Filenames
    np.save(os.path.join(save_root, "train_filenames.npy"), dataset.Y_paths[dataset.train_idxs])  # fmt: skip
    np.save(os.path.join(save_root, "test_filenames.npy"), dataset.Y_paths[dataset.test_idxs])  # fmt: skip

    # Empty context as mean over subjects
    X_null = torch.zeros_like(dataset.X[0], device=device)
    Z_null = brain_encoder.encode(X_null, torch.arange(len(X_null), device=device))
    # ( 4, 768 )
    Z_null = Z_null.mean(dim=0).unsqueeze(0).detach().cpu().numpy()
    np.save(os.path.join(save_root, "empty_context.npy"), Z_null)

    # Embeddings
    for split, datas in zip(["train", "test"], [train_set, test_set]):
        save_dir = os.path.join(save_root, split)
        os.makedirs(save_dir, exist_ok=True)

        for idx, (X, images, subject_idxs) in tqdm(enumerate(datas), total=len(datas)):
            """
            NOTE: We don't need normalization for OpenAI pretrained CLIP. MEG embeddings are
            normalized within encode method.
            """
            # image_clip = preprocess(image_clip).to(device)
            # Y_clip = clip_model.encode_image(image_clip.unsqueeze(0)).float()
            # Y_clip = Y_clip.squeeze(0).detach().cpu().numpy()
            # cprint(f"Y_clip (image): {Y_clip.shape}, {Y_clip.dtype}, Mean: {Y_clip.mean()}, Std: {Y_clip.std()}", "cyan")  # fmt: skip

            moments = autoencoder.encode_moments(images.to(device).unsqueeze(0))
            moments = moments.squeeze(0).detach().cpu().numpy()
            # cprint(f"KL_moment (image): {moments.shape}, {moments.dtype}, Mean: {moments.mean()}, Std: {moments.std()}", "cyan")  # fmt: skip
            np.save(os.path.join(save_dir, f"{idx}.npy"), moments)

            Z = brain_encoder.encode(X.to(device), subject_idxs.to(device))
            # ( 4, 768 )
            # cprint(f"Z (MEG): {Z.shape}, {Z.dtype}, Mean: {Z.mean()}, Std: {Z.std()}", "cyan")  # fmt: skip

            # NOTE: In U-ViT implementation they get multiple latents per caption, while we have multiple subjects per image.
            for i, Z_ in enumerate(Z):
                # NOTE: Unsqueezing corresponds to making the dimension of 77 in CLIP-Text.
                Z_ = Z_.unsqueeze(0).detach().cpu().numpy()
                np.save(os.path.join(save_dir, f"{idx}_{i}.npy"), Z_)


if __name__ == "__main__":
    main()
