import streamlit as st
import base64
import tempfile
import os
import json
from datetime import datetime
from mistralai import Mistral
from PIL import Image
import io

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

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Mistral OCR Processor", layout="wide")
    
    # Sidebar: Authentication for Mistral API
    api_key = st.sidebar.text_input("Mistral API Key", type="password")
    
    if not api_key:
        st.warning("Enter API key to continue")
        return
    
    # Initialize Mistral API client
    client = Mistral(api_key=api_key)
    
    # Main app interface
    st.header("Mistral OCR Processor")
    
    # Input method selection: URL, PDF Upload, or Image Upload
    input_method = st.radio("Select Input Type:", ["URL", "PDF Upload", "Image Upload"])
    
    document_source = None
    preview_content = None
    content_type = None
    
    if input_method == "URL":
        # Handle document URL input
        url = st.text_input("Document URL:")
        if url:
            document_source = {
                "type": "document_url",
                "document_url": url
            }
            preview_content = url
            content_type = "url"
    
    elif input_method == "PDF Upload":
        # Handle PDF file upload
        uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
        if uploaded_file:
            content = uploaded_file.read()
            preview_content = uploaded_file
            
            # Save the uploaded PDF temporarily for display purposes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                pdf_path = tmp.name
            
            display_pdf(pdf_path)  # Display the uploaded PDF
            
            # Prepare document source for OCR processing
            document_source = {
                "type": "document_url",
                "document_url": upload_pdf(client, content, uploaded_file.name)
            }
            content_type = "pdf"
    
    elif input_method == "Image Upload":
        # Handle image file upload
        uploaded_image = st.file_uploader("Choose Image file", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
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
    
    if document_source and st.button("Process Document"):
        # Process the document when the user clicks the button
        with st.spinner("Extracting content..."):
            try:
                ocr_response = process_ocr(client, document_source)
                
                if ocr_response and ocr_response.pages:
                    # Combine extracted text from all pages into one string
                    extracted_content = "\n\n".join(
                        [f"**Page {i+1}**\n{page.markdown}" 
                         for i, page in enumerate(ocr_response.pages)]
                    )
                    
                    # Display extracted content in Markdown format
                    st.subheader("Extracted Content")
                    st.markdown(extracted_content)
                    
                    # Prepare plain text version
                    plain_text_content = "\n\n".join(
                        [f"Page {i+1}\n{page.markdown}" 
                         for i, page in enumerate(ocr_response.pages)]
                    )
                    
                    # Prepare JSON data structure
                    json_data = {
                        "metadata": {
                            "extraction_timestamp": datetime.now().isoformat(),
                            "input_type": content_type,
                            "total_pages": len(ocr_response.pages),
                            "processor": "Mistral OCR API"
                        },
                        "content": {
                            "full_text": plain_text_content,
                            "pages": []
                        }
                    }
                    
                    # Add individual page data to JSON
                    for i, page in enumerate(ocr_response.pages):
                        page_data = {
                            "page_number": i + 1,
                            "text": page.markdown,
                            "word_count": len(page.markdown.split()) if page.markdown else 0
                        }
                        json_data["content"]["pages"].append(page_data)
                    
                    # Convert to JSON string
                    json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
                    
                    # Add download buttons for text, Markdown, and JSON formats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label="Download as Text",
                            data=plain_text_content,
                            file_name="extracted_content.txt",
                            mime="text/plain"
                        )
                    with col2:
                        st.download_button(
                            label="Download as Markdown",
                            data=extracted_content,
                            file_name="extracted_content.md",
                            mime="text/markdown"
                        )
                    with col3:
                        st.download_button(
                            label="Download as JSON",
                            data=json_content,
                            file_name="extracted_content.json",
                            mime="application/json"
                        )
                    
                    # Optional: Show raw response for debugging purposes
                    with st.expander("Raw API Response"):
                        st.json(ocr_response.model_dump())
                
                else:
                    st.warning("No content extracted.")
            
            except Exception as e:
                # Display an error message if processing fails
                st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()