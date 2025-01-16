from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from ultralytics import YOLOv10
import io
import requests
#import virtualenv
import os

#venv_dir = os.path.join(os.path.expanduser("~"), ".venv")
#virtualenv.create_environment(venv_dir)
#exec(open(os.path.join(os.path.expanduser("~"), ".venv", "Scripts", "fast_api.py")).read(), {'__file__': os.path.join(os.path.expanduser("~"), ".venv", "Scripts", "fast_api.py')})


app = FastAPI()

# Load the YOLO model (you can also include logic to download it from Hugging Face if not available)
model_path = "best.pt"
model = YOLOv10(model_path, task='detect')

@app.middleware("https")
@app.get("/")
def hello():
    return "Welcome to this fastapi"

@app.middleware("https")
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), confidence: float = 0.05):
    # Read the uploaded image file
    image_bytes = await file.read()
    
    # Open the image using PIL
    input_image = Image.open(io.BytesIO(image_bytes))
    
    # Convert PIL image to numpy array for model processing
    image_np = np.array(input_image)
    
    # Perform object detection using YOLOv10 model
    results = model(source=image_np, conf=confidence, save=False)
    
    # Plot the detected objects on the image
    result_image_np = results[0].plot()  # This is a numpy array with detections

    # Convert the result back to a PIL image
    result_image_pil = Image.fromarray(result_image_np)
    
    # Prepare the image for response by saving it to a byte stream
    img_byte_arr = io.BytesIO()
    result_image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Return the image with detections as a StreamingResponse
    return StreamingResponse(img_byte_arr, media_type="image/png")

# To run the app:
# uvicorn fastapi_app:app --reload
