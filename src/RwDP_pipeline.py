import os 
import easydict
from transformers import CLIPImageProcessor, XLMRobertaTokenizer
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import diffusers
from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import requests
import PIL
from io import BytesIO
import json
from tqdm import tqdm

import inspect
from typing import Callable, Dict, List, Optional, Union


import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
# %%
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import StableDiffusionInstructPix2PixPipeline
from torch.optim import AdamW



class StableDiffusionInstructPix2PixRegionwiseGuidancePipeline(StableDiffusionInstructPix2PixPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,):
        super().__init__(vae,text_encoder,tokenizer,unet,scheduler,safety_checker,feature_extractor, requires_safety_checker)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, 
            do_binarize=True, do_convert_grayscale=True)
    def prepare_mask_latents(
        self, mask,  batch_size, height, width, dtype, device
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)
        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        
        return mask
    def prepare_latents(self,image_latent,guidance_region_map_latent, num_iterations,  batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        """shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma"""
        noise = torch.randn_like(image_latent)
        noise2 = torch.randn_like(image_latent)
        timestep = self.scheduler.timesteps[0]
        latents = self.scheduler.add_noise(image_latent, noise, timestep)
        #print('ground truth noise scaling')
        # Initialize scaling factor as a learnable parameter
        scaling_factor = torch.tensor(1.0, requires_grad=True, device=image_latent.device)#torch.ones_like(self.scheduler.timesteps, requires_grad = True,dtype = latents.dtype)#torch.tensor(1.0, requires_grad=True, device=image_latent.device)
        scaling_factor.requires_grad_(True)

        # Define optimizer to tune the scaling factor
        optimizer = AdamW([scaling_factor], lr=0.01)
        for iteration in range(num_iterations):
            # Reset recovered_latent to noisy_latents[0]
            current_recovered_latent = latents.clone()

            # Scale the noise with the current scaling factor
            scaled_gt_noise = noise * scaling_factor

            # Ensure no gradient is retained from previous iterations
            optimizer.zero_grad()
            for t in self.scheduler.timesteps:#zip(self.scheduler.timesteps,scaling_factor):#reversed(self.scheduler.timesteps):
                # Apply the scaled noise in the step method
                current_recovered_latent = self.scheduler.scale_model_input(current_recovered_latent,t)
                current_recovered_latent = self.scheduler.step(
                        model_output=scaled_gt_noise, timestep=t, sample=current_recovered_latent,
                    return_dict=False
                    )[0]

            # Compute the L2 loss between the recovered latent and the original image_latent
            loss = torch.norm(current_recovered_latent - image_latent)

            # Backpropagate the loss to update the scaling factor
            loss.backward(retain_graph = True)

            # Update the scaling factor
            optimizer.step()

            # Print the progress
            #if iteration % 10 == 0 or iteration == num_iterations - 1:
            #    print(f"Iteration {iteration}: Loss = {loss.item()} {scaling_factor}")

        # After tuning, use the final scaling factor to get the best recovered_latent
        #final_recovered_latent = noisy_latents.clone()
        #print(noise.shape, scaling_factor.shape)
        scaled_noise = noise*scaling_factor.detach()#.unsqueeze(dim = 0) * scaling_factor.detach()
        latents = guidance_region_map_latent*latents + (1-guidance_region_map_latent)*noise2
        #print(scaled_noise.shape)
        return latents,scaled_noise

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_region_map = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be repainted according to `prompt`. Can also accept
                image latents as `image`, but if passing latents directly it is not encoded again.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_guidance_scale (`float`, *optional*, defaults to 1.5):
                Push the generated image towards the inital `image`. Image guidance scale is enabled by setting
                `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
                linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a
                value of at least `1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInstructPix2PixPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

        >>> image = download_image(img_url).resize((512, 512))

        >>> pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        ...     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "make the mountains snowy"
        >>> image = pipe(prompt=prompt, image=image).images[0]
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Check inputs
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Image latents
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
            generator,
        )

        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor
        
        if guidance_region_map is not None:
            # 6.1 Prepare mask latents
            guidance_region_map = self.mask_processor.preprocess(guidance_region_map,height=height, width=width, )
            guidance_region_map = (1-guidance_region_map)
            guidance_region_map_latent = self.prepare_mask_latents(guidance_region_map,
                        1,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                    )


        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        with torch.enable_grad():
            latents, gt_noise = self.prepare_latents(image_latents[0].unsqueeze(0)*self.vae.config.scaling_factor,
                guidance_region_map_latent,
                num_inference_steps,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        #self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else latents

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    scaled_latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False
                )[0]

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. So we need to compute the
                # predicted_original_sample here if we are using a karras style scheduler.
                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == t).nonzero()[0].item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred

                # perform guidance
                if self.do_classifier_free_guidance:
                    if guidance_region_map is not None:
                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                        #noise_pred_image = gt_noise#noise_pred_image 
                        #noise_pred_uncond = (1-guidance_region_map_latent) * noise_pred_uncond 
                        noise_pred = guidance_region_map_latent*gt_noise +                             (noise_pred_uncond + image_guidance_scale*(noise_pred_image-noise_pred_uncond)+guidance_scale*(noise_pred_text-noise_pred_image))*(1-guidance_region_map_latent)
                        """another working version : noise_pred = guidance_region_map_latent*gt_noise +                             guidance_scale*(noise_pred_text-noise_pred_image)*(1-guidance_region_map_latent)"""
                    else:
                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + self.guidance_scale * (noise_pred_text - noise_pred_image)
                            + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )
                #print(noise_pred.shape)
                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - latents) / (-sigma)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    image_latents = callback_outputs.pop("image_latents", image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents/self.vae.config.scaling_factor, return_dict=False)[0]#
            has_nsfw_concept =None
            #image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
