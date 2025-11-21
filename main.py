
# AI for this: Salesforce/blip-image-captioning-large

#pip install pillow
#pip install transformers torch torchvision pillow
#python.exe -m pip install --upgrade pip
#pip install transformers pillow
#pip install hf_xet

# ---------------------------------------FOR IMAGE TO TEXT-------------------------------------------------

from transformers import pipeline
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# --------------------------------------FOR TEXT GENERATING------------------------------------------------

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

#-----------------------

# ---------------------------------------IMAGE TO TEXT-------------------------------------------------
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-large")

image = Image.open("Images/noveny2.jpg")

# ezzel a prompttal kényszerítjük, hogy fajnevet próbáljon mondani
prompt = "a photo of a plant species:"
inputs = processor(images=image, text=prompt, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print("BLIP növényfelismerés:", caption)

# -------------------------------------TEXT GENERATING--------------------------------------------------

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