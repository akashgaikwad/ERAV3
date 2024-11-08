from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import string

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Custom stopwords list
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
    'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'can', 'just', 'should', 'now'
}

def tokenize(text):
    """Simple tokenizer function"""
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Split on whitespace
    return text.split()

def process_text(text: str, preprocessing_type: str) -> dict:
    """Process text based on selected preprocessing type"""
    if preprocessing_type == "remove_stopwords":
        tokens = tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in STOP_WORDS]
        processed_text = ' '.join(filtered_tokens)
        return {
            "original_text": text,
            "processed_text": processed_text,
            "removed_words": len(tokens) - len(filtered_tokens)
        }
    
    elif preprocessing_type == "lowercase":
        processed_text = text.lower()
        return {
            "original_text": text,
            "processed_text": processed_text,
            "operation": "converted to lowercase"
        }
    
    elif preprocessing_type == "remove_punctuation":
        processed_text = text.translate(str.maketrans("", "", string.punctuation))
        return {
            "original_text": text,
            "processed_text": processed_text,
            "operation": "removed punctuation"
        }
    
    elif preprocessing_type == "tokenize":
        tokens = tokenize(text)
        return {
            "original_text": text,
            "tokens": tokens,
            "token_count": len(tokens)
        }
    
    else:
        raise ValueError(f"Unsupported preprocessing type: {preprocessing_type}")

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    with open("index.html") as file:
        return file.read()

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        if file.content_type.startswith('text/'):
            text = content.decode('utf-8')
            return {"content": text}
            
        elif file.content_type.startswith("image/"):
            # Save image and return path
            file_location = f"static/uploads/{file.filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(content)
            return FileResponse(file_location)
            
        elif file.content_type.startswith("audio/"):
            # Save audio and return URL
            file_location = f"static/uploads/{file.filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(content)
            return {"audio_url": f"/static/uploads/{file.filename}"}
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preprocess")
async def preprocess_file(
    file: UploadFile = File(...),
    preprocessing_type: str = Form(...)
):
    try:
        content = await file.read()
        
        if file.content_type.startswith('text/'):
            text = content.decode('utf-8')
            result = process_text(text, preprocessing_type)
            return JSONResponse(result)
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Currently only text preprocessing is supported"
            )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ensure uploads directory exists
os.makedirs("static/uploads", exist_ok=True)
