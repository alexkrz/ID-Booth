import torch

from diffusers import DDPMScheduler, StableDiffusionPipeline


base_model = "stabilityai/stable-diffusion-2-1-base"
lora_checkpoint = "checkpoints/ID-Booth/ID_1/checkpoint-31-6400"  # Download or train your own

prompt = "face portrait photo of male sks person, city street background"
negative_prompt = "cartoon, render, illustration, painting, drawing, black and white, bad body proportions, landscape"

pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda:0")
pipe.scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
pipe.load_lora_weights(lora_checkpoint)

image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=30, guidance_scale=5.0).images[0]

image.save(f"results/ID_1_{prompt}.png")
