from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from ultralytics import YOLOv10, YOLO
from ultralytics.nn.tasks import YOLOv10DetectionModel
import io
import requests
import os
import json
import cloudinary
import cloudinary.uploader
import tempfile
import torch
import torch
from functools import partial
from ultralytics.nn.modules.conv import Conv


# Setting up Cloudinary Credentials
cloudinary.config(
    cloud_name="dmtd60ln7",
    api_key="736887161144449",
    api_secret="gTClh1QjelFwhQolG9ZnIYf9KXw",
    secure=True,
)


#Explicitly allow necessary YOLOv10 layers
torch.serialization.add_safe_globals([Conv])

#Override `torch.load()` to force `weights_only=False`
original_torch_load = torch.load  # Keep a reference to the original function

def custom_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # Force weights_only=False
    return original_torch_load(*args, **kwargs)

#Apply the monkey patch
torch.load = custom_torch_load

#Now, load the YOLOv10 model
model_path = "best.pt"
model = YOLOv10(model_path, task='detect')

#Restore the original torch.load after loading the model
torch.load = original_torch_load

app = FastAPI()

@app.get("/")
def hello():
    return "Welcome to this fastapi"

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), confidence: float = 0.05):
    global results, cloudinary_image

    # Read the uploaded image file
    image_bytes = await file.read()
    
    # Open the image using PIL
    input_image = Image.open(io.BytesIO(image_bytes))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)

        # Save the image to the temporary directory
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(image_bytes)

        # Upload the image file to Cloudinary
        cloudinary_image = cloudinary.uploader.upload(temp_file_path, unique_filename=False, overwrite=True)
    
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

@app.get("/json_response")
def get_json_response() -> dict:
    # Get the results of te detection from the detect_object function
    detect_objects()

    # Convert the results to json string
    for result in results:
        json_result_string = json.loads(result.tojson())
        
    # Extracting the url from Cloudinary
    image_url = cloudinary_image["secure_url"]

    # Put everything together and add the key name for every dectected object
    final_dict = {}
    for i in range(len(json_result_string)):
        final_dict[f"Detection-{i}"] = json_result_string[i]

        # Add the image url to the final dictionary
        final_dict["image_url"] = image_url
    
    json_dumped_result = json.dumps(final_dict, indent=3)
    
    # Convert the JSON response
    return final_dict

# To run the app:
# uvicorn fastapi_app:app --reload
