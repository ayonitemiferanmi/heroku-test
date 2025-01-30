from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import numpy as np
from ultralytics import YOLOv10
from ultralytics.nn.tasks import YOLOv10DetectionModel
import io
import os
import json
import cloudinary
import cloudinary.uploader
import tempfile
import torch

# Setting up Cloudinary Credentials
cloudinary.config(
    cloud_name="dmtd60ln7",
    api_key="736887161144449",
    api_secret="gTClh1QjelFwhQolG9ZnIYf9KXw",
    secure=True,
)

torch.serialization.add_safe_globals([YOLOv10DetectionModel])

app = FastAPI()

# Load the YOLO model
model_path = "best.pt"
model = YOLOv10(model=model_path, task="detect")

# Store latest detection results (temporary)
latest_results = {}

async def process_image(file: UploadFile, confidence: float):
    """Processes image: uploads to Cloudinary, runs YOLO detection, and returns results."""
    # Read uploaded image
    image_bytes = await file.read()
    input_image = Image.open(io.BytesIO(image_bytes))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)

        # Save image temporarily
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(image_bytes)

        # Upload image to Cloudinary
        cloudinary_image = cloudinary.uploader.upload(temp_file_path, unique_filename=True, overwrite=True)

    image_url = cloudinary_image["secure_url"]

    # Convert to numpy array for model processing
    image_np = np.array(input_image)

    # Perform object detection
    results = model(source=image_np, conf=confidence, save=False)

    return results, image_url

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), confidence: float = 0.05):
    """Uploads image, detects objects, returns only the processed image."""
    results, image_url = await process_image(file, confidence)

    # Store latest results (overwrites previous)
    global latest_results
    latest_results = {
        "detections": [json.loads(result.tojson()) for result in results],
        "image_url": image_url
    }

    # Plot detected objects
    result_image_np = results[0].plot()

    # Convert to PIL image
    result_image_pil = Image.fromarray(result_image_np)

    # Convert to byte stream
    img_byte_arr = io.BytesIO()
    result_image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Return processed image
    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.get("/json_response")
async def get_json_response():
    """Returns detection results in JSON format."""
    if not latest_results:
        return JSONResponse(content={"error": "No detection results available"}, status_code=404)

    return JSONResponse(content=latest_results)
