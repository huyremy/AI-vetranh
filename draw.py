from diffusers import StableDiffusionPipeline
import torch
model_id = "path-to-your-trained-model" # đường dẫn đến model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
prompt = "A dog eating bones" # thích vẽ cái gì thì điền vào đây
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-eating.png")
