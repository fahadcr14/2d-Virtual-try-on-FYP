# filename: app.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import os
import uuid
from API import generate_viton_output  # <- adjust if needed

app = FastAPI()

@app.post("/generate/")
async def viton_generate(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    height: int = Form(512),
    width: int = Form(384),
    base_model_path: str = Form("runwayml/stable-diffusion-inpainting"),
    resume_path: str = Form("zhengchong/CatVTON"),
    mixed_precision: str = Form("fp16"),
    num_inference_steps: int = Form(100),
    guidance_scale: float = Form(2.5),
    seed: int = Form(555),
    repaint: bool = Form(False),
    concat_eval_results: bool = Form(True),
    cloth_type: str = Form("upper")
):
    # Create a temp folder
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Save uploaded files
    person_image_path = os.path.join(temp_dir, f"{uuid.uuid4()}_person.png")
    cloth_image_path = os.path.join(temp_dir, f"{uuid.uuid4()}_cloth.png")

    with open(person_image_path, "wb") as buffer:
        shutil.copyfileobj(person_image.file, buffer)
    with open(cloth_image_path, "wb") as buffer:
        shutil.copyfileobj(cloth_image.file, buffer)

    # Run inference
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        output_path = generate_viton_output(
            person_image=person_image_path,
            cloth_image=cloth_image_path,
            output_dir=output_dir,
            height=height,
            width=width,
            base_model_path=base_model_path,
            resume_path=resume_path,
            mixed_precision=mixed_precision,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            repaint=repaint,
            concat_eval_results=concat_eval_results,
            cloth_type=cloth_type
        )

        return FileResponse(output_path, media_type="image/png", filename=os.path.basename(output_path))

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Clean up uploaded files
        if os.path.exists(person_image_path):
            os.remove(person_image_path)
        if os.path.exists(cloth_image_path):
            os.remove(cloth_image_path)

import uvicorn
import threading

def run_app():
    uvicorn.run("fapi:app", host="0.0.0.0", port=8000)

threading.Thread(target=run_app).start()
