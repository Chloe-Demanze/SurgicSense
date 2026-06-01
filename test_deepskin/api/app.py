import os
import uuid
import shutil
import zipfile
import asyncio
from typing import Optional

from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse

from api.my_env import my_env
from src.wound_image import WoundImage

TEMPLATES = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "templates")
TEMP_DIR = os.path.join("output", "api", "temp")
MODELS_DIR = os.path.join("output", "models")
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
EXPECTED_FORMATS = {
    "segmentation_mask": "save_segmentation_mask",
    "segmentation_semantic": "save_segmentation_semantic",
    "mask_wound": "save_mask_wound",
    "mask_peri_wound": "save_mask_peri_wound",
    "masked_wound": "save_masked_wound",
    "masked_peri_wound": "save_masked_peri_wound",
    "pwat_estimation": "save_pwat_estimation",
}


def gen_id():
    return str(uuid.uuid4())


async def auto_delete_file(file_path: str, delay: float):
    await asyncio.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)


async def auto_delete_dir(dir_path: str, delay: float):
    await asyncio.sleep(delay)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def _seed_demo_models():
    """Pre-populate demo models from the input/ folder."""
    input_dir = "input"
    if not os.path.exists(input_dir):
        return
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in VALID_EXTENSIONS:
            model_name = os.path.splitext(f)[0]
            model_dir = os.path.join(MODELS_DIR, model_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                shutil.copy(os.path.join(input_dir, f), os.path.join(model_dir, f))


def _find_texture(model_dir: str) -> Optional[str]:
    """Return path to the first non-segmented texture image in model_dir."""
    for f in sorted(os.listdir(model_dir)):
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
            if not f.startswith("segmented_"):
                return os.path.join(model_dir, f)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    _seed_demo_models()
    yield
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


# ------------------------------------------------------------------
# Original endpoints
# ------------------------------------------------------------------

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/valid_extensions")
async def get_valid_extensions():
    return list(VALID_EXTENSIONS)


@app.get("/expected_formats")
async def get_expected_formats():
    return list(EXPECTED_FORMATS.keys())


@app.get("/upload")
async def get_upload():
    return FileResponse(os.path.join(TEMPLATES, 'upload.html'))


@app.post("/upload")
async def upload_image(expected_format: str,
                       file: UploadFile = File(...)) -> FileResponse:
    """Upload and process an image based on the expected format."""
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in VALID_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid image format. Use one of: {await get_valid_extensions()}.")

    if expected_format not in EXPECTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Invalid expected format. Use one of: {await get_expected_formats()}")

    file_uuid = gen_id()
    file_path = os.path.join(TEMP_DIR, f"{file_uuid}{file_ext}")
    file_new_path = os.path.join(TEMP_DIR, f"{file_uuid}_processed{file_ext}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    wi = WoundImage(image_path=file_path, logging=my_env.is_dev())
    predicted_pwat = wi.get_predicted_pwat()
    getattr(wi, EXPECTED_FORMATS[expected_format])(file_new_path)

    asyncio.create_task(auto_delete_file(file_path, delay=1))
    asyncio.create_task(auto_delete_file(file_new_path, delay=1))

    return FileResponse(file_new_path, media_type=file.content_type, headers={
                        "predicted_pwat": str(predicted_pwat)})


@app.get("/upload/pwat")
async def get_upload_pwat():
    return FileResponse(os.path.join(TEMPLATES, 'upload_pwat.html'))


@app.post("/upload/pwat")
async def pwat_from_image(file: UploadFile = File(...)) -> JSONResponse:
    """Upload an image and return the predicted PWAT score."""
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in VALID_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid image format. Use one of: {await get_valid_extensions()}.")

    file_path = os.path.join(TEMP_DIR, f"{gen_id()}{file_ext}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    wi = WoundImage(image_path=file_path, logging=my_env.is_dev())
    predicted_pwat = wi.get_predicted_pwat()

    asyncio.create_task(auto_delete_file(file_path, delay=1))

    return JSONResponse(content={"predicted_pwat": predicted_pwat})


# ------------------------------------------------------------------
# Frontend workflow endpoints (/api/...)
# ------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    """Return the list of available scan model names."""
    if not os.path.exists(MODELS_DIR):
        return []
    return sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])


@app.get("/api/models/{model_name}/{filename}")
async def serve_model_file(model_name: str, filename: str):
    """Serve a static file (texture, OBJ, MTL) from a model's directory."""
    file_path = os.path.join(MODELS_DIR, model_name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.post("/api/preview")
async def preview_model(model_name: str = Form(...)):
    """Return the texture image URL for a given model."""
    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail="Model not found")
    texture = _find_texture(model_dir)
    if not texture:
        raise HTTPException(status_code=404, detail="No texture found")
    return {"texture_url": f"/api/models/{model_name}/{os.path.basename(texture)}"}


@app.post("/api/upload")
async def upload_model(
    model_name: str = Form(...),
    obj_file: UploadFile = File(...),
    texture_file: UploadFile = File(...),
    mtl_file: UploadFile = File(...)
):
    """Upload OBJ + texture + MTL files as a new scan model."""
    model_dir = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_dir):
        raise HTTPException(status_code=409, detail="Model already exists")
    os.makedirs(model_dir)
    for f in [obj_file, texture_file, mtl_file]:
        dest = os.path.join(model_dir, f.filename)
        with open(dest, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
    return {"message": "Upload successful"}


@app.post("/api/upload-zip")
async def upload_zip(zip_file: UploadFile = File(...)):
    """Upload a ZIP archive containing a scan model's files."""
    zip_path = os.path.join(TEMP_DIR, f"{gen_id()}.zip")
    with open(zip_path, "wb") as buf:
        shutil.copyfileobj(zip_file.file, buf)
    model_name = os.path.splitext(zip_file.filename)[0]
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(model_dir)
    os.remove(zip_path)
    return {"message": "ZIP uploaded successfully"}


@app.post("/api/segment")
async def segment_model(model_name: str = Form(...), method: str = Form("deepskin")):
    """Run Deepskin wound segmentation on a model's texture image."""
    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail="Model not found")
    texture_path = _find_texture(model_dir)
    if not texture_path:
        raise HTTPException(status_code=404, detail="No texture found for segmentation")
    result_filename = "segmented_" + os.path.basename(texture_path)
    result_path = os.path.join(model_dir, result_filename)
    wi = WoundImage(image_path=texture_path, logging=my_env.is_dev())
    wi.save_segmentation_semantic(result_path)
    return {"texture_url": f"/api/models/{model_name}/{result_filename}"}
