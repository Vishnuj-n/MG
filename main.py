import streamlit as st
import requests
import google.generativeai as genai
import json
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS



class meme:
    @staticmethod
    def encode_sequential(text):
        replacements = [
            ('_', '__'), ('-', '--'), (' ', '_'), ('?', '~q'),
            ('&', '~a'), ('%', '~p'), ('#', '~h'), ('/', '~s'),
            ('\\', '~b'), ('<', '~l'), ('>', '~g'), ('"', "''"),
            ('\n', '~n'), ('__', '___')
        ]
        encoded_text = text
        for original, replacement in replacements:
            encoded_text = encoded_text.replace(original, replacement)
        return encoded_text
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

    @staticmethod
    def template_chooser(image_prompt):
        # Fallback method using Gemini generation (not used if vector search works)
        try:
            with open("template.json", "r") as file:
                templates = json.load(file)
            template_names = [template["name"] for template in templates]
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = f"Which meme template fits the following description best? {image_prompt}\nOptions: {', '.join(template_names)}"
            response = model.generate_content(prompt)
            response_text = response.text.lower()
            for template in templates:
                if template["name"].lower() in response_text:
                    return template["id"]
            return "distracted-boyfriend"
        except Exception as e:
            st.error(f"Error selecting template: {e}")
            return "distracted-boyfriend"
    
    @staticmethod
    def find_similar_templates(user_prompt, k=3):
        """
        Load all meme templates from template.json,
        create a description for each (combining the name and keywords),
        build a vector store, and then search for the top k templates
        similar to the user prompt.
        """
        try:
            with open("template.json", "r") as file:
                templates = json.load(file)
            texts = []
            metadatas = []
            for template in templates:
                # Create a text description using the meme name and its keywords (if any)
                description = template["name"]
                if "keywords" in template and template["keywords"]:
                    description += " " + " ".join(template["keywords"])
                texts.append(description)
                metadatas.append(template)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
            results = vector_store.similarity_search(user_prompt, k=k)
            similar_templates = [doc.metadata for doc in results]
            return similar_templates
        except Exception as e:
            st.error(f"Error finding similar templates: {e}")
            return []

    @staticmethod
    def create_meme(upper_text, lower_text, image_url):
        try:
            up_text = meme.encode_sequential(upper_text)
            down_text = meme.encode_sequential(lower_text)
            meme_url = f"https://api.memegen.link/images/{image_url}/{up_text}/{down_text}.png"
            response = requests.head(meme_url)
            response.raise_for_status()
            return meme_url
        except Exception as e:
            st.error(f"Error creating meme: {e}")
            return None

    @staticmethod
    def get_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store

def main():
    st.title("Meme Generator")

    # Add descriptive help text
    st.markdown("""
    ### Instructions:
    1. Enter a description of the meme you want to create
    2. Enter the text you want at the top of the meme
    3. Enter the text you want at the bottom of the meme
    """)

    # Add input fields with placeholder text and validation
    image_prompt = st.text_input(
        "Enter image description:", 
        placeholder="Example: A person making a tough choice between two options"
    )
    upper_text = st.text_input(
        "Enter upper text:", 
        placeholder="Text that will appear at the top of the meme"
    )
    lower_text = st.text_input(
        "Enter lower text:", 
        placeholder="Text that will appear at the bottom of the meme"
    )

    # Validate inputs before processing
    if st.button("Generate Memes"):
        if not image_prompt:
            st.error("Please enter an image description.")
        elif not upper_text and not lower_text:
            st.warning("At least one text field (upper or lower) must be filled.")
        else:
            similar_templates = meme.find_similar_templates(image_prompt, k=3)
            
            if similar_templates:
                st.subheader("Generated Memes:")
                cols = st.columns(min(3, len(similar_templates)))
                
                for i, template in enumerate(similar_templates):
                    with cols[i % 3]:
                        template_id = template["id"]
                        template_name = template["name"]
                        
                        meme_url = meme.create_meme(upper_text, lower_text, template_id)
                        if meme_url:
                            st.image(meme_url, caption=f"Template: {template_name}")
                            st.write(f"[Open Meme URL]({meme_url})")
            else:
                st.warning("No similar templates found. Please try a different description.")

if __name__ == "__main__":
    main()
