from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import shutil
import uuid
from fastapi.concurrency import run_in_threadpool
from critique_engine import analyze_squat # Assuming your logic is here

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-and-critique")
async def upload_and_critique(file: UploadFile = File(...)):
    # 1. Validation
    if not file.filename.endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid video format.")

    # 2. Save file with unique ID to prevent collisions
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 3. Run heavy analysis in a separate thread to avoid blocking
        # This calls your function: analyze(video_path) -> returns JSON
        analysis_result = await run_in_threadpool(analyze_squat, video_path)
        return analysis_result
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # 4. Clean up the file after analysis to save disk space
        if os.path.exists(video_path):
            os.remove(video_path)