import streamlit as st
import base64
import tempfile
import os
import json
import re
from datetime import datetime
from mistralai import Mistral
from PIL import Image
import io
from dotenv import load_dotenv

def upload_pdf(client, content, filename):
    """
    Uploads a PDF to Mistral's API and retrieves a signed URL for processing.
    
    Args:
        client (Mistral): Mistral API client instance.
        content (bytes): The content of the PDF file.
        filename (str): The name of the PDF file.

    Returns:
        str: Signed URL for the uploaded PDF.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, "wb") as tmp:
            tmp.write(content)
        
        try:
            with open(temp_path, "rb") as file_obj:
                file_upload = client.files.upload(
                    file={"file_name": filename, "content": file_obj},
                    purpose="ocr"
                )
            
            signed_url = client.files.get_signed_url(file_id=file_upload.id)
            return signed_url.url
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def process_ocr(client, document_source):
    """
    Processes a document using Mistral's OCR API.

    Args:
        client (Mistral): Mistral API client instance.
        document_source (dict): The source of the document (URL or image).

    Returns:
        OCRResponse: The response from Mistral's OCR API.
    """
    return client.ocr.process(
        model="mistral-ocr-latest",
        document=document_source,
        include_image_base64=True
    )

def extract_resume_data(client, extracted_text):
    """
    Uses Mistral AI to extract structured resume data from the extracted text.
    
    Args:
        client (Mistral): Mistral API client instance.
        extracted_text (str): The extracted text from the resume.
    
    Returns:
        dict: Structured resume data in JSON format.
    """
    
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
        
        # Try to parse JSON, handling potential markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
            
        return json.loads(response_text)
        
    except Exception as e:
        st.error(f"Error extracting resume data: {str(e)}")
        return create_empty_resume_structure()

def create_empty_resume_structure():
    """
    Creates an empty resume structure with all required fields.
    
    Returns:
        dict: Empty resume structure.
    """
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

def display_pdf(file):
    """
    Displays a PDF in Streamlit using an iframe.

    Args:
        file (str): Path to the PDF file.
    """
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

def display_resume_summary(resume_data):
    """
    Displays a summary of the extracted resume data.
    
    Args:
        resume_data (dict): Structured resume data.
    """
    st.subheader("📋 Resume Summary")
    
    # Basics
    if resume_data["basics"]["name"]:
        st.write(f"**Name:** {resume_data['basics']['name']}")
    if resume_data["basics"]["email"]:
        st.write(f"**Email:** {resume_data['basics']['email']}")
    if resume_data["basics"]["phone"]:
        st.write(f"**Phone:** {resume_data['basics']['phone']}")
    
    # Work Experience
    if resume_data["work"]:
        st.write(f"**Work Experience:** {len(resume_data['work'])} positions")
    
    # Education
    if resume_data["education"]:
        st.write(f"**Education:** {len(resume_data['education'])} entries")
    
    # Skills
    total_skills = (len(resume_data["skills"]["technical"]) + 
                   len(resume_data["skills"]["professional"]) + 
                   len(resume_data["skills"]["languages_programming"]) + 
                   len(resume_data["skills"]["tools"]))
    if total_skills > 0:
        st.write(f"**Skills:** {total_skills} total skills")
    
    # Projects
    if resume_data["projects"]:
        st.write(f"**Projects:** {len(resume_data['projects'])} projects")
    
    # Other sections
    sections = ["volunteer", "awards", "certificates", "publications", "languages", "interests", "references"]
    for section in sections:
        if resume_data[section]:
            st.write(f"**{section.title()}:** {len(resume_data[section])} entries")

def create_empty_resume_structure():
    """
    Creates an empty resume structure with all required fields.
    
    Returns:
        dict: Empty resume structure.
    """
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

def main():
    """
    Main function to run the Resume Parser Streamlit app.
    """
    st.set_page_config(page_title="AI Resume Parser", layout="wide", page_icon="📄")
    
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("API_KEY_NAME")
    
    if not api_key:
        st.error("API key not found. Please set API_KEY_NAME in your environment variables.")
        st.stop()
    
    # Initialize Mistral API client
    client = Mistral(api_key=api_key)
    
    # Main app interface
    st.title("AI Resume Parser")
    st.markdown("Upload a resume (PDF or image) and get structured JSON data automatically!")
    
    # Input method selection
    input_method = st.radio("Select Input Type:", ["PDF Upload", "Image Upload", "URL"], horizontal=True)
    
    document_source = None
    content_type = None
    filename = None
    
    if input_method == "URL":
        url = st.text_input("🔗 Enter Document URL:")
        if url:
            document_source = {
                "type": "document_url",
                "document_url": url
            }
            content_type = "url"
            filename = "resume_from_url"
    
    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
        if uploaded_file:
            content = uploaded_file.read()
            filename = uploaded_file.name.replace('.pdf', '')
            
            # Display the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                pdf_path = tmp.name
            
            with st.expander("View Uploaded PDF"):
                display_pdf(pdf_path)
            
            # Prepare document source for OCR processing
            document_source = {
                "type": "document_url",
                "document_url": upload_pdf(client, content, uploaded_file.name)
            }
            content_type = "pdf"
    
    elif input_method == "Image Upload":
        uploaded_image = st.file_uploader("Choose Image file", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            filename = uploaded_image.name.rsplit('.', 1)[0]
            
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="📷 Uploaded Resume Image", use_container_width=True)
            
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare document source for OCR processing
            document_source = {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{img_str}"
            }
            content_type = "image"
    
    # Auto-process when document is uploaded
    if document_source:
        with st.spinner("🔄 Processing resume and extracting data..."):
            try:
                # Step 1: Extract text using OCR
                st.info("Step 1: Extracting text from document...")
                ocr_response = process_ocr(client, document_source)
                
                if ocr_response and ocr_response.pages:
                    # Combine extracted text from all pages
                    extracted_text = "\n\n".join([page.markdown for page in ocr_response.pages])
                    
                    # Step 2: Parse resume data using AI
                    st.info("Step 2: Analyzing and structuring resume data...")
                    resume_data = extract_resume_data(client, extracted_text)
                    
                    # Add metadata
                    resume_json = {
                        "metadata": {
                            "extraction_timestamp": datetime.now().isoformat(),
                            "input_type": content_type,
                            "filename": filename,
                            "total_pages": len(ocr_response.pages),
                            "processor": "Mistral AI Resume Parser"
                        },
                        "resume": resume_data
                    }
                    
                    # Display results
                    st.success("✅ Resume processed successfully!")
                    
                    # Create two columns for display
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        display_resume_summary(resume_data)
                    
                    with col2:
                        st.subheader("Download Options")
                        
                        # Generate filename
                        safe_filename = filename.replace(' ', '_').replace('.', '_') if filename else "resume"
                        json_filename = f"{safe_filename}_parsed.json"
                        
                        # JSON download button
                        json_content = json.dumps(resume_json, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download Resume JSON",
                            data=json_content,
                            file_name=json_filename,
                            mime="application/json",
                            type="primary"
                        )
                        
                        # Raw text download
                        st.download_button(
                            label="📄 Download Raw Text",
                            data=extracted_text,
                            file_name=f"{safe_filename}_raw_text.txt",
                            mime="text/plain"
                        )
                    
                    # Show detailed JSON structure
                    with st.expander("🔍 View Detailed JSON Structure"):
                        st.json(resume_json)
                    
                    # Show extraction statistics
                    with st.expander("Extraction Statistics"):
                        stats = {
                            "Total sections found": len([k for k, v in resume_data.items() if v and k != "basics"]),
                            "Work experiences": len(resume_data.get("work", [])),
                            "Education entries": len(resume_data.get("education", [])),
                            "Skills categories": len([k for k, v in resume_data.get("skills", {}).items() if v]),
                            "Projects": len(resume_data.get("projects", [])),
                            "Certificates": len(resume_data.get("certificates", [])),
                            "Languages": len(resume_data.get("languages", [])),
                            "Total words extracted": len(extracted_text.split())
                        }
                        
                        for key, value in stats.items():
                            st.metric(key, value)
                
                else:
                    st.error("No content could be extracted from the document.")
            
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                st.info("Please try with a different file or check your API key.")
    
    else:
        st.info("👆 Please upload a resume file or provide a URL to get started.")
        
        # Show feature list
        st.markdown("### What this parser extracts:")
        
        features = [
            "**Basics** - Personal information (name, email, summary)",
            "**Work** - Work experience with details",
            "**Education** - Educational background",
            " **Skills** - Technical and professional skills",
            " **Projects** - Portfolio projects",
            " **Volunteer** - Volunteer work",
            " **Awards** - Awards and recognition",
            " **Certificates** - Professional certifications",
            " **Publications** - Published works",
            " **Languages** - Spoken languages",
            " **Interests** - Personal interests",
            " **References** - Professional references"
        ]
        
        for feature in features:
            st.markdown(feature)

if __name__ == "__main__":
    main()