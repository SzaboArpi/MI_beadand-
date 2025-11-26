
# AI for this: Salesforce/blip-image-captioning-large |  GPT2LMHeadModel | 

#pip install pillow
#pip install transformers torch torchvision pillow
#python.exe -m pip install --upgrade pip
#pip install transformers pillow
#pip install diffusers transformers accelerate open_clip_torch
#pip install --upgrade diffusers transformers accelerate torch
# pip install opencv-python

# ---------------------------------------FOR IMAGE TO TEXT-------------------------------------------------

from transformers import pipeline
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# --------------------------------------FOR TEXT GENERATING------------------------------------------------

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

# -------------------------------------FOR TEXT TO VIDEO-----------------------------------------------------

import open_clip
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

#________________________________________________________________________________________________________________#

# --------------------------------------- IMAGE TO TEXT -------------------------------------------------
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-large")

image = Image.open("Images/noveny2.jpg")

# ezzel a prompttal kényszerítjük, hogy fajnevet próbáljon mondani
prompt = "a photo of a plant species:"
inputs = processor(images=image, text=prompt, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print("BLIP növényfelismerés:", caption)

# ------------------------------------- TEXT GENERATING --------------------------------------------------

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model_gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

generator = pipeline(
    "text-generation",
    model=model_gpt2,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

prompt = f"{caption} is a plant that"

result = generator(
    prompt,
    max_length=100,
    num_return_sequences=1,
    temperature=0.85,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)[0]["generated_text"]

# Remove the prompt part and clean up
description = result[len(prompt):].strip()
print("\nDescription:")
print(description)

# -------------------------------------------- TEXT TO VIDEO ------------------------------------------------

print(f"\nGenerálok videót erről: {caption}")

# Prompt a növény kinövéséről
prompt = f"Time-lapse of a {caption} growing from seed in soil, realistic nature close-up, sprout emerging, leaves unfolding, soft sunlight, high quality"
negative_prompt = "blurry, low quality, text, watermark, deformed"

# Legacy modell: nincs fp16 hiba, kisebb és stabilabb
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b-legacy",  # ← Ez a kulcs: legacy verzió
)

# Scheduler optimalizálás (gyorsabb generálás)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# GPU optimalizálás (ha van)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.enable_vae_slicing()        # További optimalizálás

# Videó generálása (1–3 perc)
print("Videó generálása... (légy türelmes!)")
video_frames = pipe(
    prompt,
    num_inference_steps=25,   # 25–50: minőség vs. sebesség
    height=256,
    width=256,
    num_frames=16,            # 16 frame ≈ 2 mp
    guidance_scale=9.0,
    negative_prompt=negative_prompt
).frames[0]  # [0] a nested list miatt

# Mentés MP4-be (8 fps)
safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in caption.lower())[:40]
output_file = f"growing_{safe_name}.mp4"
export_to_video(video_frames, output_file, fps=8)

print(f"\nVideó kész! Mentve: {os.path.abspath(output_file)}")