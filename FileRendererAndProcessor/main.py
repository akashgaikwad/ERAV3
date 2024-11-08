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
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Currently only text augmentation is supported"
            )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ensure uploads directory exists
os.makedirs("static/uploads", exist_ok=True)
