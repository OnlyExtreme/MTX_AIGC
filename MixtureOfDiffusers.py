from enum import Enum
import inspect
from typing import List, Optional, Tuple, Union

import torch

from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import LMSDiscreteScheduler

from extrasmixin import StableDiffusionExtrasMixin


class MixtureOfDiffusersPipeline(DiffusionPipeline, StableDiffusionExtrasMixin):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: Union[DDIMScheduler, PNDMScheduler],
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae = vae,
            text_encoder = text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[List[str]]],
        num_inference_steps: Optional[int] = 20,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        seed: Optional[int] = None,
        tile_height: Optional[int] = 512,
        tile_width: Optional[int] = 512,
        tile_row_overlap: Optional[int] = 256,
        tile_col_overlap: Optional[int] = 256,
        guidance_scale_tiles: Optional[List[List[float]]] = None,
        seed_tiles: Optional[List[List[int]]] = None,
        seed_tiles_mode: Optional[Union[str, List[List[str]]]] = "full",
        seed_reroll_regions: Optional[List[Tuple[int, int, int, int, int]]] = None,
        cpu_vae: Optional[bool] = False,
    ):
        
        grid_rows = len(prompt)
        grid_cols = len(prompt[0])
        batch_size = 1

        # Original noisy latents
        height = tile_height + (grid_rows - 1) * (tile_height - tile_row_overlap)
        width = tile_width + (grid_cols - 1) * (tile_width - tile_col_overlap)
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = torch.randn(latents_shape, generator = generator, device = self.device)

        # Prepare Scheduler
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # Get Prompts text embaddings
        prompt_input = [
            [
                self.tokenizer(
                    col,
                    padding = "max_length",
                    max_length = self.tokenizer.model_max_length,
                    truncation = True,
                    return_tensors = "pt",
                )
                for col in row
            ]
            for row in prompt
        ]
        text_embeddings = [
            [
                self.text_encoder(col.input_ids.to("cuda"))[0]
                for col in row
            ]
            for row in prompt_input
        ]

        # Set weight of classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        # Get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            for i in range(grid_rows):
                for j in range(grid_cols):
                    max_length = prompt_input[i][j].input_ids.shape[-1]
                    unconditional_input = self.tokenizer(
                        [""] * batch_size,
                        padding = "max_length", max_length = max_length, return_tensors = "pt"
                    )
                    unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to("cuda"))[0]
        
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Weights for smooth transition between region
        tile_weights = self._gaussian_weights(tile_width, tile_height, batch_size)

        # Diffusion main stage
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # Diffuse each region
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    px_row_init, px_row_end, px_col_init, px_col_end = tile2latent(row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap)
                    tile_latents = latents[:, :, px_row_init:px_row_end, px_col_init:px_col_end]
                    # Expand for classifier free guidance
                    latent_model_input = torch.cat([tile_latents] * 2) if do_classifier_free_guidance else tile_latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = latent_model_input.to(torch.float16)
                    
                    # Predict noise
                    noise_preds = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[row][col])["sample"]
                    # Guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        guidance = guidance_scale if guidance_scale_tiles is None or guidance_scale_tiles[row][col] is None else guidance_scale_tiles[row][col]
                        # z = z_guidance + strength * (z_guidance - z_no_guidance)
                        noise_pred_tile = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
                        noise_preds_row.append(noise_pred_tile)
                noise_preds.append(noise_preds_row)

            # Apply noise prediction
            noise_pred = torch.zeros(latents.shape, device = self.device)
            contributors = torch.zeros(latents.shape, device = self.device)

            # Sum up contributions
            for row in range(grid_rows):
                for col in range(grid_cols):
                    px_row_init, px_row_end, px_col_init, px_col_end = tile2latent(row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap)
                    noise_pred[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += noise_preds[row][col] * tile_weights
                    contributors[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += tile_weights
            # Process transition areas
            noise_pred /= contributors

            # Calculate previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]
        
        # Scale up and decode the latents with autoencoder (vae)
        image = self.decode_latents(latents, cpu_vae)

        return {"sample": image}
    
    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        # Generate 2 dimensional gaussian weight for smooth transition
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width // 8
        latent_height = tile_height // 8

        var = .01
        mid = (latent_width - 1) / 2
        # Gaussian distribution: exp(-(x-mean)/(2*sigma*sigma))/2/sqrt(2*pi*sigma)
        x_probs = [exp(-(x-mid)*(x-mid)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        mid = (latent_height - 1) / 2
        y_probs = [exp(-(y-mid)*(y-mid)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        # Make weight into 2 dimension by vector "outer" operation
        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.in_channels, 1, 1))
    
def tile2pixel(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap):
    # Calculate the range of pixels affected by the given tile
    px_row_init = 0 if tile_row == 0 else tile_row * (tile_height - tile_row_overlap)
    px_row_end = px_row_init + tile_height
    px_col_init = 0 if tile_col == 0 else tile_col * (tile_width - tile_col_overlap)
    px_col_end = px_col_init + tile_width
    return px_row_init, px_row_end, px_col_init, px_col_end

def pixel2latent(px_row_init, px_row_end, px_col_init, px_col_end):
    return px_row_init // 8, px_row_end // 8, px_col_init // 8, px_col_end // 8
    
def tile2latent(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap):
    # Calculate the range of pixels in latents affected by the given tile
    px_row_init, px_row_end, px_col_init, px_col_end = tile2pixel(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap)
    return pixel2latent(px_row_init, px_row_end, px_col_init, px_col_end)