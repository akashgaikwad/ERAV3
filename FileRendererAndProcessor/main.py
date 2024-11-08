from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import string
import random
from nltk.corpus import wordnet
import nltk
import ssl
from deep_translator import GoogleTranslator
import time
from PIL import Image
import torchvision.transforms as transforms
import io
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal

# Add this SSL context fix for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data with error handling
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Also download Open Multilingual WordNet

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

def back_translate(text: str, intermediate_lang='es') -> str:
    """
    Translate text to an intermediate language and back to English
    Using delay to avoid Google Translate API rate limits
    """
    try:
        # Translate to intermediate language
        time.sleep(1)  # Add delay to avoid rate limiting
        translator = GoogleTranslator(source='en', target=intermediate_lang)
        intermediate = translator.translate(text)
        
        # Translate back to English
        time.sleep(1)  # Add delay to avoid rate limiting
        translator = GoogleTranslator(source=intermediate_lang, target='en')
        back_translated = translator.translate(intermediate)
        
        return back_translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def get_random_words(text: str, n: int) -> list:
    """Get n random words from the text that aren't stopwords"""
    words = [word for word in text.split() if word.lower() not in STOP_WORDS]
    return random.sample(words, min(n, len(words)))

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

def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():  # Don't include the word itself
                synonyms.add(lemma.name())
    return list(synonyms)

def augment_text(text: str, augmentation_type: str) -> dict:
    """Augment text based on selected augmentation type"""
    if augmentation_type == "synonym_replace":
        words = text.split()
        num_words = len(words)
        # Replace 30% of replaceable words with synonyms
        num_to_replace = max(1, int(num_words * 0.3))
        
        # Keep track of which words have synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if len(get_synonyms(word)) > 0:
                replaceable_indices.append(i)
        
        # Randomly select words to replace
        if replaceable_indices:
            indices_to_replace = random.sample(replaceable_indices, 
                                            min(num_to_replace, len(replaceable_indices)))
            
            for idx in indices_to_replace:
                synonyms = get_synonyms(words[idx])
                if synonyms:
                    words[idx] = random.choice(synonyms)
        
        augmented_text = ' '.join(words)
        return {
            "original_text": text,
            "augmented_text": augmented_text,
            "words_replaced": len(indices_to_replace) if replaceable_indices else 0
        }
    
    elif augmentation_type == "back_translation":
        # Translate to Spanish and back to English
        augmented_text = back_translate(text)
        return {
            "original_text": text,
            "augmented_text": augmented_text,
            "operation": "back translation (via Spanish)"
        }
    
    elif augmentation_type == "random_insert":
        words = text.split()
        num_words = len(words)
        num_to_insert = max(1, int(num_words * 0.15))  # Insert 15% new words
        
        # Get random words from the original text
        words_to_insert = get_random_words(text, num_to_insert)
        
        # Insert words at random positions
        for word in words_to_insert:
            insert_position = random.randint(0, len(words))
            words.insert(insert_position, word)
        
        augmented_text = ' '.join(words)
        return {
            "original_text": text,
            "augmented_text": augmented_text,
            "words_inserted": len(words_to_insert)
        }
    
    else:
        raise ValueError(f"Unsupported augmentation type: {augmentation_type}")

# Add these image processing functions
def process_image(image: Image.Image, preprocessing_type: str) -> dict:
    """Process image based on selected preprocessing type"""
    try:
        if preprocessing_type == "resize":
            # Resize to 224x224 (standard size)
            processed_img = transforms.Resize((224, 224))(image)
            return {
                "processed_image": processed_img,
                "operation": "resized to 224x224"
            }
        
        elif preprocessing_type == "normalize":
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            processed_img = transform(image)
            # Convert back to PIL for display
            processed_img = transforms.ToPILImage()(processed_img)
            return {
                "processed_image": processed_img,
                "operation": "normalized"
            }
        
        elif preprocessing_type == "grayscale":
            processed_img = transforms.Grayscale(num_output_channels=1)(image)
            return {
                "processed_image": processed_img,
                "operation": "converted to grayscale"
            }
        
        elif preprocessing_type == "crop":
            # Center crop to 224x224
            processed_img = transforms.CenterCrop(224)(image)
            return {
                "processed_image": processed_img,
                "operation": "center cropped to 224x224"
            }
        
        else:
            raise ValueError(f"Unsupported preprocessing type: {preprocessing_type}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def augment_image(image: Image.Image, augmentation_type: str) -> dict:
    """Augment image based on selected augmentation type"""
    try:
        if augmentation_type == "rotate":
            # Random rotation between -30 and 30 degrees
            angle = random.uniform(-30, 30)
            augmented_img = transforms.functional.rotate(image, angle)
            return {
                "augmented_image": augmented_img,
                "operation": f"rotated by {angle:.1f} degrees"
            }
        
        elif augmentation_type == "flip":
            # Random horizontal flip
            augmented_img = transforms.functional.hflip(image)
            return {
                "augmented_image": augmented_img,
                "operation": "horizontally flipped"
            }
        
        elif augmentation_type == "blur":
            # Gaussian blur
            blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
            augmented_img = blur(image)
            return {
                "augmented_image": augmented_img,
                "operation": "gaussian blur applied"
            }
        
        elif augmentation_type == "brightness":
            # Adjust brightness
            factor = random.uniform(0.5, 1.5)
            augmented_img = transforms.functional.adjust_brightness(image, factor)
            return {
                "augmented_image": augmented_img,
                "operation": f"brightness adjusted by factor {factor:.2f}"
            }
        
        else:
            raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Simple audio processing functions using only scipy
def process_audio(audio_data: bytes, preprocessing_type: str) -> dict:
    """Process audio based on selected preprocessing type"""
    try:
        # Save the uploaded audio temporarily
        temp_path = os.path.join("static/uploads", "temp_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(audio_data)
        
        # Read the audio file
        sample_rate, audio = wavfile.read(temp_path)
        
        # Convert to float32 for processing
        audio = audio.astype(np.float32) / 32767.0
        
        if preprocessing_type == "normalize_audio":
            # Normalize audio
            processed_audio = audio / np.max(np.abs(audio))
            operation = "audio normalized"
            
        elif preprocessing_type == "trim_silence":
            # Simple silence trimming
            threshold = 0.01
            mask = np.abs(audio) > threshold
            processed_audio = audio[mask]
            operation = "silence trimmed"
            
        elif preprocessing_type == "resample":
            # Resample to half the original sample rate
            target_sr = sample_rate // 2
            processed_audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))
            sample_rate = target_sr
            operation = f"resampled to {target_sr}Hz"
            
        elif preprocessing_type == "noise_reduction":
            # Simple noise reduction using a low-pass filter
            b, a = signal.butter(3, 0.1)
            processed_audio = signal.filtfilt(b, a, audio)
            operation = "noise reduced"
            
        else:
            raise ValueError(f"Unsupported preprocessing type: {preprocessing_type}")
        
        # Convert back to int16 and save
        processed_audio = (processed_audio * 32767).astype(np.int16)
        output_path = os.path.join("static/uploads", f"processed_audio_{int(time.time())}.wav")
        wavfile.write(output_path, sample_rate, processed_audio)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "processed_audio_url": f"/static/uploads/{os.path.basename(output_path)}",
            "operation": operation
        }
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

def augment_audio(audio_data: bytes, augmentation_type: str) -> dict:
    """Augment audio based on selected augmentation type"""
    try:
        # Save the uploaded audio temporarily
        temp_path = os.path.join("static/uploads", "temp_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(audio_data)
        
        # Read the audio file
        sample_rate, audio = wavfile.read(temp_path)
        audio = audio.astype(np.float32) / 32767.0
        
        if augmentation_type == "time_stretch":
            # Simple time stretching (slower/faster)
            factor = np.random.uniform(0.8, 1.2)
            augmented_audio = signal.resample(audio, int(len(audio) * factor))
            operation = f"time stretched by factor {factor:.2f}"
            
        elif augmentation_type == "pitch_shift":
            # Simple pitch shifting using resampling
            factor = np.random.uniform(0.8, 1.2)
            augmented_audio = signal.resample(audio, int(len(audio) * factor))
            augmented_audio = signal.resample(augmented_audio, len(audio))
            operation = "pitch shifted"
            
        elif augmentation_type == "add_noise":
            # Add random noise
            noise_level = 0.005
            noise = np.random.normal(0, noise_level, len(audio))
            augmented_audio = audio + noise
            operation = "random noise added"
            
        elif augmentation_type == "reverb":
            # Simple reverb effect
            delay = int(sample_rate * 0.1)  # 100ms delay
            decay = 0.3
            impulse = np.zeros(delay)
            impulse[0] = 1
            impulse[delay-1] = decay
            augmented_audio = signal.convolve(audio, impulse, mode='same')
            operation = "reverb added"
            
        else:
            raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
        
        # Convert back to int16 and save
        augmented_audio = np.clip(augmented_audio, -1, 1)
        augmented_audio = (augmented_audio * 32767).astype(np.int16)
        output_path = os.path.join("static/uploads", f"augmented_audio_{int(time.time())}.wav")
        wavfile.write(output_path, sample_rate, augmented_audio)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "augmented_audio_url": f"/static/uploads/{os.path.basename(output_path)}",
            "operation": operation
        }
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

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
            
        elif file.content_type.startswith('image/'):
            # Image preprocessing
            image = Image.open(io.BytesIO(content))
            result = process_image(image, preprocessing_type)
            
            # Save processed image and return path
            output_path = f"static/uploads/processed_{file.filename}"
            result["processed_image"].save(output_path)
            
            return {
                "processed_image_url": f"/static/uploads/processed_{file.filename}",
                "operation": result["operation"]
            }
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Currently only text and image preprocessing is supported"
            )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augment")
async def augment_file(
    file: UploadFile = File(...),
    augmentation_type: str = Form(...)
):
    try:
        content = await file.read()
        
        if file.content_type.startswith('text/'):
            text = content.decode('utf-8')
            result = augment_text(text, augmentation_type)
            return JSONResponse(result)
            
        elif file.content_type.startswith('image/'):
            # Image augmentation
            image = Image.open(io.BytesIO(content))
            result = augment_image(image, augmentation_type)
            
            # Save augmented image and return path
            output_path = f"static/uploads/augmented_{file.filename}"
            result["augmented_image"].save(output_path)
            
            return {
                "augmented_image_url": f"/static/uploads/augmented_{file.filename}",
                "operation": result["operation"]
            }
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Currently only text and image augmentation is supported"
            )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ensure uploads directory exists
os.makedirs("static/uploads", exist_ok=True)
