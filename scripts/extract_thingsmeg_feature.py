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

from nd.datasets.things_meg import ThingsMEGDatabase
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
        "U-ViT/assets/stable-diffusion/autoencoder_kl.pth"
    ).to(device)

    # CLIP-Vision ViT-B/32
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model = clip_model.eval().to(device)

    # CLIP-MEG
    brain_encoder = BrainEncoder(
        args,
        subject_names=dataset.subject_names,
        layout=eval(args.layout),
        vq=args.vq,
        blocks=args.blocks,
        downsample=args.downsample,
        temporal_aggregation=args.temporal_aggregation,
    ).to(device)
    brain_encoder.load_state_dict(
        torch.load(
            os.path.join(get_run_dir(args), "brain_encoder_best.pt"),
            map_location=device,
        )
    )
    brain_encoder.eval()

    # -----------------
    #      Extract
    # -----------------
    for split, datas in zip(["train", "test"], [train_set, test_set]):
        save_dir = f"assets/datasets/thingsmeg_features/{split}"
        os.makedirs(save_dir, exist_ok=True)

        for idx, data in tqdm(enumerate(datas)):
            X, image_clip, image_sd, subject_idx = data

            Z = brain_encoder.encode(X.to(device), subject_idx.to(device))
            Z = Z.detach().cpu().numpy()
            cprint(f"Z (MEG): {Z.shape}, {Z.dtype}, Mean: {Z.mean()}, Std: {Z.std()}", "cyan")  # fmt: skip

            image_clip = preprocess(image_clip).to(device)
            Y_clip = clip_model.encode_image(image_clip.unsqueeze(0)).float()
            Y_clip /= Y_clip.norm(dim=-1, keepdim=True)
            Y_clip = Y_clip.squeeze(0).detach().cpu().numpy()
            cprint(f"Y_clip (image): {Y_clip.shape}, {Y_clip.dtype}, Mean: {Y_clip.mean()}, Std: {Y_clip.std()}", "cyan")  # fmt: skip

            image_sd = torch.tensor(image_sd, device=device).unsqueeze(0)
            moments = autoencoder(image_sd, fn="encode_moments").squeeze(0)
            moments = moments.detach().cpu().numpy()
            cprint(f"Y_sd (image): {moments.shape}, {moments.dtype}, Mean: {moments.mean()}, Std: {moments.std()}", "cyan")  # fmt: skip

            sys.exit()

            # ------------------------------------------

            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            moments = autoencoder(x, fn="encode_moments").squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f"{idx}.npy"), moments)

            latent = clip.encode(captions)
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f"{idx}_{i}.npy"), c)


if __name__ == "__main__":
    main()
