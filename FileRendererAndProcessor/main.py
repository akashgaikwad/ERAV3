from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Add this middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory with custom headers
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Endpoint to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    with open("index.html") as file:
        return file.read()

# File upload endpoint
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_location = f"uploads/{file.filename}"
    
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    
    if file.content_type.startswith("text/"):
        with open(file_location, "r") as f:
            content = f.read()
        return {"content": content}
    
    elif file.content_type.startswith("image/"):
        return FileResponse(file_location)
    
    elif file.content_type.startswith("audio/"):
        return {"audio_url": f"/static/uploads/{file.filename}"}
    
    elif file.content_type == "model/3d":
        return {"message": "3D model uploaded"}
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
