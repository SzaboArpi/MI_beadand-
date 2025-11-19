#IMAGE TO TEXT
#AI for this: Salesforce/blip-image-captioning-large
# Use a pipeline as a high-level helper
from transformers import pipeline
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVision2Seq

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
# Load model directly


processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-large")

#példa
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)
#text = "A picture of"
#inputs = processor(images=image, text=text, return_tensors="pt")
#outputs = model(**inputs)