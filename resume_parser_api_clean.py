from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import tempfile
import os
import json
import base64
from datetime import datetime
from mistralai import Mistral
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Resume Parser API",
    description="Upload resume files (PDF/Image) and get structured JSON data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mistral client
API_KEY = os.environ.get("API_KEY_NAME")
if not API_KEY:
    raise ValueError("API_KEY_NAME environment variable is required")

client = Mistral(api_key=API_KEY)

def upload_pdf_to_mistral(content, filename):
    """Upload PDF to Mistral's API and get signed URL - matches Streamlit app exactly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, "wb") as tmp:
            tmp.write(content)
        
        try:
            with open(temp_path, "rb") as file_obj:
                # Use exact same format as working Streamlit app
                file_upload = client.files.upload(
                    file={"file_name": filename, "content": file_obj},
                    purpose="ocr"
                )
            
            signed_url = client.files.get_signed_url(file_id=file_upload.id)
            return signed_url.url
        except Exception as e:
            print(f"PDF upload error: {str(e)}")
            print(f"Error type: {type(e)}")
            # Let's see what the actual error is
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to upload PDF: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def process_ocr(document_source):
    """Process document using Mistral's OCR API."""
    return client.ocr.process(
        model="mistral-ocr-latest",
        document=document_source,
        include_image_base64=True
    )

def extract_resume_data(extracted_text):
    """Extract structured resume data using Mistral AI - matches Streamlit app exactly."""
    
    prompt = f"""
    You are an expert resume parser. Extract the following information from this resume text and return it as a structured JSON object. 
    If any section is not found, include it with empty values but maintain the structure.
    
    Resume text:
    {extracted_text}
    
    Please extract and structure the information into this exact JSON format:
    {{
        "basics": {{
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "website": "",
            "linkedin": "",
            "summary": ""
        }},
        "work": [
            {{
                "company": "",
                "position": "",
                "startDate": "",
                "endDate": "",
                "description": "",
                "highlights": []
            }}
        ],
        "education": [
            {{
                "institution": "",
                "degree": "",
                "field": "",
                "startDate": "",
                "endDate": "",
                "gpa": "",
                "description": ""
            }}
        ],
        "skills": {{
            "technical": [],
            "professional": [],
            "languages_programming": [],
            "tools": []
        }},
        "projects": [
            {{
                "name": "",
                "description": "",
                "technologies": [],
                "startDate": "",
                "endDate": "",
                "url": "",
                "highlights": []
            }}
        ],
        "volunteer": [
            {{
                "organization": "",
                "position": "",
                "startDate": "",
                "endDate": "",
                "description": "",
                "highlights": []
            }}
        ],
        "awards": [
            {{
                "title": "",
                "date": "",
                "awarder": "",
                "description": ""
            }}
        ],
        "certificates": [
            {{
                "name": "",
                "issuer": "",
                "date": "",
                "url": "",
                "description": ""
            }}
        ],
        "publications": [
            {{
                "title": "",
                "publisher": "",
                "date": "",
                "url": "",
                "description": ""
            }}
        ],
        "languages": [
            {{
                "language": "",
                "fluency": ""
            }}
        ],
        "interests": [
            {{
                "name": "",
                "keywords": []
            }}
        ],
        "references": [
            {{
                "name": "",
                "position": "",
                "company": "",
                "email": "",
                "phone": "",
                "relationship": ""
            }}
        ]
    }}
    
    Instructions:
    1. Extract all available information accurately
    2. Use consistent date formats (YYYY-MM or YYYY-MM-DD)
    3. For arrays, include all relevant items found
    4. If information is not available, use empty strings or empty arrays
    5. Be thorough in extracting highlights and descriptions
    6. Return ONLY the JSON object, no additional text
    """
    
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        # Extract JSON from response
        response_text = response.choices[0].message.content.strip()
        
        # Clean up response text (same as Streamlit app)
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
            
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error extracting resume data: {str(e)}")
        return create_empty_resume_structure()

def create_empty_resume_structure():
    """Create empty resume structure - matches Streamlit app exactly."""
    return {
        "basics": {
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "website": "",
            "linkedin": "",
            "summary": ""
        },
        "work": [],
        "education": [],
        "skills": {
            "technical": [],
            "professional": [],
            "languages_programming": [],
            "tools": []
        },
        "projects": [],
        "volunteer": [],
        "awards": [],
        "certificates": [],
        "publications": [],
        "languages": [],
        "interests": [],
        "references": []
    }

@app.get("/")
async def root():
    """API information."""
    return {
        "message": "AI Resume Parser API",
        "status": "active",
        "version": "1.0.0",
        "description": "Upload resume files (PDF/Image) and get structured JSON data - matches Streamlit app functionality",
        "endpoints": {
            "parse_resume": "/parse-resume (POST) - Upload resume file and get structured JSON",
            "health": "/health (GET) - API health check"
        }
    }

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    """
    Parse resume from uploaded file (PDF or Image).
    Returns structured JSON with resume data - exactly like Streamlit app.
    """
    try:
        print(f"Processing file: {file.filename}")
        
        # Validate file type
        allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
        file_extension = os.path.splitext(file.filename.lower())[1]
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file extension: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=422, detail="File appears to be empty")
        
        filename = file.filename.rsplit('.', 1)[0] if '.' in file.filename else file.filename
        
        # Determine file type and prepare document source
        if file_extension == '.pdf':
            print("Processing as PDF...")
            document_url = upload_pdf_to_mistral(content, file.filename)
            document_source = {
                "type": "document_url",
                "document_url": document_url
            }
            content_type = "pdf"
        else:
            print("Processing as Image...")
            image = Image.open(io.BytesIO(content))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            document_source = {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{img_str}"
            }
            content_type = "image"
        
        # Step 1: Extract text using OCR (same as Streamlit app)
        print("Step 1: Extracting text from document...")
        ocr_response = process_ocr(document_source)
        
        if not ocr_response or not ocr_response.pages:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document")
        
        # Combine extracted text from all pages (same as Streamlit app)
        extracted_text = "\n\n".join([page.markdown for page in ocr_response.pages])
        
        # Step 2: Parse resume data using AI (same as Streamlit app)
        print("Step 2: Analyzing and structuring resume data...")
        resume_data = extract_resume_data(extracted_text)
        
        # Prepare final response (same structure as Streamlit app)
        result = {
            "metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "input_type": content_type,
                "filename": filename,
                "total_pages": len(ocr_response.pages),
                "processor": "Mistral AI Resume Parser"
            },
            "resume": resume_data
        }
        
        print("Resume processing completed successfully!")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_key_configured": bool(API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)