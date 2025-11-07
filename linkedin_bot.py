import os
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from typing import List
from io import BytesIO
from pypdf import PdfReader

# --- Pydantic Schemas ---

class LinkedInPost(BaseModel):
    """Defines the structure for a single generated post."""
    tone: str = Field(description="The tone of the post (e.g., 'Celebratory', 'Reflective').")
    content: str = Field(description="The full body of the LinkedIn post. Must adhere to the word limit.")
    hashtags: List[str] = Field(description="A list of 3 to 5 relevant hashtags for the post.")

class LinkedInPostsList(BaseModel):
    """The wrapper model for the final list of posts."""
    posts: List[LinkedInPost]
    
# --- File Processing Function ---

@st.cache_data
def extract_text_from_pdf(uploaded_file_bytes):
    """Extracts all text content from an uploaded PDF file stream."""
    try:
        pdf_reader = PdfReader(BytesIO(uploaded_file_bytes))
        text = ""
        for page in pdf_reader.pages:
            # Safely extract text
            text += page.extract_text() or "" 
        return text
    except Exception:
        # Return empty string if extraction fails
        return ""

# --- Core Generation Logic ---

def generate_linkedin_posts(user_context: str, max_words: int, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile = None):
    """
    Calls the Gemini API to generate 5 LinkedIn posts based on input and constraints.
    """
    # 1. API Key Check (Robust)
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

    if not api_key:
        st.error("🚨 API Key not found. Please set 'GEMINI_API_KEY' in your environment or in `.streamlit/secrets.toml`.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        contents = [
            "You are an expert LinkedIn copywriter specializing in career achievements. Your task is to analyze the provided context (and image/document data, if available) and then generate 5 unique LinkedIn posts."
        ]
        
        # --- 2. Handle File Input (Multimodal or Text Extraction) ---
        file_context = ""
        if uploaded_file is not None:
            file_type = uploaded_file.type
            uploaded_file_bytes = uploaded_file.read() 
            
            if file_type.startswith('image/'):
                st.info(f"Using **Image** ({uploaded_file.name}) for analysis.")
                image = Image.open(BytesIO(uploaded_file_bytes))
                contents.append(image)
                file_context = "I've uploaded an image (e.g., certificate, photo) for content analysis."
                
            elif file_type == 'application/pdf':
                st.info(f"Using **PDF Text Extraction** ({uploaded_file.name}) for context.")
                pdf_text = extract_text_from_pdf(uploaded_file_bytes)
                if pdf_text:
                    # Truncate to keep the prompt size reasonable
                    truncated_text = pdf_text[:4000] 
                    file_context = f"The following text was extracted from an uploaded PDF document: \n\n---START PDF TEXT---\n{truncated_text}\n---END PDF TEXT---"
                else:
                    st.warning("Could not find usable text in the PDF. Relying only on the primary context.")
            
            else:
                st.warning(f"Unsupported file type: {file_type}. Please use image or PDF.")

        # --- 3. Construct the Final Prompt ---
        
        tones = ["Educational/Informative", "Celebratory/Enthusiastic", "Reflective/Vulnerable", "Professional/Direct", "Humorous/Relatable"]
        tones_list = ", ".join(tones)
        
        final_context = f"{file_context}\n\n**Primary User Context:** {user_context}"
        
        user_prompt_template = f"""
        **Total Context:** {final_context}
        
        **Task:** Generate 5 distinct posts about this event/achievement, one for each of the following tones:
        {tones_list}
        
        **CRITICAL CONSTRAINT: Each post MUST NOT exceed {max_words} words.**
        
        **Instructions:**
        1. Ensure each post is suitable for a professional LinkedIn audience.
        2. Use relevant emojis and mention the main achievement/context.
        3. Return the result in the required JSON schema using the 'LinkedInPostsList' model.
        """
        contents.append(user_prompt_template)
        
        # --- 4. Execute the API call ---
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents, 
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LinkedInPostsList,
            )
        )
        
        return LinkedInPostsList.model_validate_json(response.text)

    except APIError as e:
        st.error(f"Gemini API Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Streamlit GUI Layout ---

def main():
    """Defines the Streamlit application layout and user interaction."""
    st.set_page_config(page_title="Multi-File LinkedIn Post Generator", layout="wide")
    st.title("📝 AI LinkedIn Post Generator (Multi-Format)")
    st.markdown("---")

    # --- Input Section ---
    st.header("Provide Context")
    
    # 1. Text Context Input (Optional)
    user_context = st.text_area(
        "📝 Primary Context (e.g., job update, key struggle, exciting detail):",
        height=100,
        placeholder="e.g., 'Excited about my promotion to manager! This comes after 3 years of hard work and team mentorship.'",
        key="context_input"
    )

    # 2. File Upload (Optional: .jpg, .png, .pdf)
    uploaded_file = st.file_uploader(
        "📁 Upload File (Optional: .jpg, .png, .pdf):", 
        type=['png', 'jpg', 'jpeg', 'pdf'],
        key="file_uploader"
    )

    # 3. Word Limit Input
    max_words = st.slider(
        "📏 Set Maximum Word Limit for Posts:",
        min_value=50,
        max_value=250,
        value=120,
        step=10,
        key="word_limit"
    )

    st.markdown("---")
    
    # --- Generation Button and Output ---
    st.header("Generate Content")
    
    if st.button("✨ Generate 5 Posts", type="primary"):
        
        # CRITICAL VALIDATION CHECK: Minimum 1 input required
        if not user_context and not uploaded_file:
            st.error("🚨 You must provide at least one form of input: **Text Context** OR an **Image/PDF** file.")
            return

        with st.spinner("Analyzing data and generating posts with different tones..."):
            # Call the core logic function
            posts_list = generate_linkedin_posts(user_context, max_words, uploaded_file)

        if posts_list:
            st.success("✅ Content Generated Successfully!")
            st.header("Your Custom LinkedIn Posts")
            
            # Display results in columns/expanders
            cols = st.columns(2)
            
            for i, post in enumerate(posts_list.posts):
                col = cols[i % 2]
                
                with col.expander(f"**{i+1}. {post.tone.upper()}** (Target: {max_words} words)", expanded=True):
                    
                    st.code(post.content, language='markdown')
                    
                    # Display hashtags below the content
                    hashtag_str = ' '.join([f"#{h}" for h in post.hashtags])
                    st.caption(f"**Hashtags:** {hashtag_str}")
                    
                    # Text area for easy copy-paste
                    st.text_area("Copy Content:", value=post.content + "\n\n" + hashtag_str, height=200, label_visibility="collapsed")


if __name__ == "__main__":
    main()