import gradio as gr
import ollama
import json
import base64
import os
import subprocess
import logging
import re
import ast
import traceback
import threading
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import concurrent.futures
from PIL import Image
from io import BytesIO
from pydantic import BaseModel

# For document processing
from PyPDF2 import PdfReader
from docx import Document
import pyttsx3
import speech_recognition as sr

# For RAG capabilities
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_API_URL = "http://localhost:11434"

# Create an Ollama client
client = ollama.Client(host=OLLAMA_API_URL)

# Global variables
chat_history: List[Dict[str, str]] = []
markdown_history: List[str] = []
current_markdown_index: int = 0
sessions: Dict[str, Tuple[List[Dict[str, str]], List[str]]] = {}
current_session: Optional[str] = None
code_versions: List[str] = []

class ImageAnalysisSchema(BaseModel):
    """Schema for image analysis structured output"""
    summary: str
    objects: List[Dict[str, Any]]
    scene_type: str
    colors: List[str]
    text_content: Optional[str] = None

def get_available_models() -> List[str]:
    """Get list of available models from Ollama"""
    try:
        # Check if Ollama is running
        models_response = client.list()
        
        # Handle different response formats from different Ollama versions
        if 'models' in models_response:
            # New format
            return [model.get('name', model.get('model', 'unknown')) for model in models_response['models']]
        else:
            # Old format
            return [model.get('name', model.get('model', 'unknown')) for model in models_response]
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        # Return a placeholder that won't cause further errors
        return ["ollama-model-placeholder"]

def generate_text(
    model: str, 
    prompt: str, 
    max_length: int, 
    temperature: float, 
    top_k: int, 
    top_p: float,
    num_sequences: int, 
    image: Optional[str] = None, 
    context: Optional[str] = None,
    format: Optional[Dict] = None
) -> str:
    """Generate text using Ollama API with improved handling for multimodal and structured outputs"""
    full_prompt = f"{context}\n\n{prompt}" if context else prompt

    options = {
        'model': model,
        'prompt': full_prompt,
        'stream': False,
        'options': {
            'num_predict': max_length,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
        }
    }

    # Handle image for multimodal models
    if image:
        try:
            with open(image, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Try to use the newer Ollama API for images with chat first
            try:
                messages = [
                    {
                        'role': 'user',
                        'content': full_prompt,
                        'images': [img_base64]
                    }
                ]
                
                # Use chat API for multimodal
                response = client.chat(
                    model=model,
                    messages=messages,
                    format=format,
                    options={
                        'temperature': temperature,
                        'top_k': top_k,
                        'top_p': top_p,
                        'num_predict': max_length
                    }
                )
                return response['message']['content']
            except Exception as e:
                logger.warning(f"Chat API for multimodal failed: {e}, trying generate API")
                # Fallback to older generate API for images
                response = client.generate(
                    model=model,
                    prompt=full_prompt,
                    images=[img_base64],
                    options={
                        'temperature': temperature,
                        'top_k': top_k,
                        'top_p': top_p,
                        'num_predict': max_length
                    }
                )
                return response['response']
        except Exception as e:
            logger.error(f"Error in multimodal generation: {e}")
            return f"An error occurred with image processing: {str(e)}"

    try:
        responses = []
        for _ in range(num_sequences):
            # Use format if provided for structured outputs
            if format:
                response = client.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': full_prompt}],
                    format=format,
                    options={
                        'temperature': temperature,
                        'top_k': top_k,
                        'top_p': top_p,
                        'num_predict': max_length
                    }
                )
                generated_text = response['message']['content'].strip()
            else:
                response = client.generate(**options)
                generated_text = response['response'].strip()
                
            responses.append(generated_text)

        return "\n\n--- New Sequence ---\n\n".join(responses)
    except Exception as e:
        logger.error(f"Error in generate_text: {e}")
        return f"An error occurred: {str(e)}"

def extract_text_from_document(file) -> Optional[str]:
    """Extract text from uploaded documents (PDF, DOCX, TXT)"""
    if file is None:
        return None

    try:
        # Check if it's a string path or file-like object
        filepath = getattr(file, 'name', file)
        
        if isinstance(filepath, str):
            if filepath.endswith('.pdf'):
                reader = PdfReader(filepath)
                return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif filepath.endswith('.docx'):
                doc = Document(filepath)
                return "\n".join(paragraph.text for paragraph in doc.paragraphs)
            elif filepath.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return f"Unsupported file format: {filepath}"
        else:
            # If we get here, it's probably a file-like object
            return "Unable to process file. Please try uploading again."
    except Exception as e:
        logger.error(f"Error extracting text from document: {e}")
        return f"Error processing document: {str(e)}"

def get_available_voices() -> List[str]:
    """Get available text-to-speech voices"""
    engine = pyttsx3.init()
    return [voice.name for voice in engine.getProperty('voices')]

def text_to_speech(text: str, voice_name: str) -> Optional[str]:
    """Convert text to speech using the selected voice"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        selected_voice = next((voice for voice in voices if voice.name == voice_name), voices[0])
        engine.setProperty('voice', selected_voice.id)

        output_file = "output.mp3"
        engine.save_to_file(text, output_file)
        engine.runAndWait()

        return output_file
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return None

def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
    
    Returns:
        The sum of the two numbers
    """
    return a + b

def get_weather(location: str) -> str:
    """
    Get the weather for a location.
    
    Args:
        location: The city or place to get weather for
    
    Returns:
        A description of the weather
    """
    # This is a mock function - in a real app, you would call a weather API
    return f"The weather in {location} is currently sunny with a temperature of 72°F."

def function_calling_handler(model: str, prompt: str) -> str:
    """Handle function calling with the Ollama API"""
    # Define available tools/functions
    tools = [add_two_numbers, get_weather]
    
    # Function registry
    available_functions = {
        "add_two_numbers": add_two_numbers,
        "get_weather": get_weather
    }
    
    try:
        # Initial request with tools
        messages = [{"role": "user", "content": prompt}]
        response = client.chat(
            model=model,
            messages=messages,
            tools=tools
        )
        
        # Check if a tool was called
        if response["message"].get("tool_calls"):
            for tool_call in response["message"]["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = tool_call["function"]["arguments"]
                
                # Convert string args to dict if needed
                if isinstance(function_args, str):
                    function_args = json.loads(function_args)
                
                # Call the function
                if function_to_call := available_functions.get(function_name):
                    function_response = function_to_call(**function_args)
                    
                    # Add function response to messages
                    messages.append(response["message"])
                    messages.append({
                        "role": "tool", 
                        "content": str(function_response),
                        "name": function_name
                    })
                    
                    # Get final response
                    final_response = client.chat(model=model, messages=messages)
                    return f"Function called: {function_name}\nArguments: {function_args}\nResult: {function_response}\n\nFinal response: {final_response['message']['content']}"
            
        # No tool called, return original response
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error in function calling: {e}")
        return f"An error occurred during function calling: {str(e)}"

def generate_with_context(
    main_model: str,
    coding_model: str,
    prompt: str,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    num_sequences: int,
    image,
    document,
    mode: str,
    voice_name: str,
    generate_voice: bool,
    json_format: bool = False,
    function_calling: bool = False
):
    """Generate response with context, handling different modes and features"""
    global chat_history, markdown_history, current_markdown_index
    
    # Extract context from document(s)
    context = None
    if document:
        # Check if it's a list or a single document
        if isinstance(document, list):
            # Process multiple documents
            doc_texts = []
            for doc in document:
                doc_text = extract_text_from_document(doc)
                if doc_text:
                    doc_texts.append(doc_text)
            if doc_texts:
                context = "\n\n".join(doc_texts)
        else:
            # Process a single document
            context = extract_text_from_document(document)
    
    # Add chat history to context
    if chat_history:
        previous_conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        context = f"{context}\n\nChat History:\n{previous_conversation}" if context else previous_conversation

    # Different modes of operation
    if mode == "Function Calling" and function_calling:
        # Handle function calling mode
        main_response = function_calling_handler(main_model, prompt)
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": main_response})
        
        audio_output = None
        if generate_voice:
            audio_output = text_to_speech(main_response, voice_name)

        return (
            chat_history_to_string(),
            "",
            chat_history_to_string(),
            audio_output,
            gr.update(value=""),
            gr.update(value=""),
        )
        
    elif mode == "Coding":
        # First generate explanation with main model
        explanation_prompt = f"Context: {context}\n\nUser Request: {prompt}\n\nProvide an explanation of the proposed changes and how to use them. Do not include any code in this response."
        explanation = generate_text(main_model, explanation_prompt, max_length, temperature, top_k, top_p, 1)

        # Then generate code with coding model
        coding_prompt = f"Generate Python code based on the following request. Only output code with proper comments. Do not include any explanations outside of code comments.\n\nContext: {context}\n\nUser Request: {prompt}\n\nMain Model Explanation: {explanation}"
        code_response = generate_text(coding_model, coding_prompt, max_length, temperature, top_k, top_p, 1)

        # Update chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": explanation})
        markdown_history.append(code_response)
        current_markdown_index = len(markdown_history) - 1
        code_versions.append(code_response)
        
        return (
            chat_history_to_string(),
            code_response,
            chat_history_to_string(),
            None,
            gr.update(value=code_response),
            gr.update(value=""),
        )
        
    elif mode == "Image Analysis" and image:
        # Use structured output schema for image analysis
        format = ImageAnalysisSchema.model_json_schema() if json_format else None
        image_prompt = f"Analyze this image in detail: {prompt}" if prompt else "Analyze this image and describe what you see in detail."
        
        main_response = generate_text(
            main_model, image_prompt, max_length, temperature, top_k, top_p, 1, image, context, format
        )
        
        chat_history.append({"role": "user", "content": image_prompt})
        chat_history.append({"role": "assistant", "content": main_response})

        audio_output = None
        if generate_voice:
            audio_output = text_to_speech(main_response, voice_name)

        return (
            chat_history_to_string(),
            "",
            chat_history_to_string(),
            audio_output,
            gr.update(value=""),
            gr.update(value=""),
        )
        
    else:  # Default chat mode
        format = None
        if json_format and mode == "Structured Output":
            # Basic schema for generic structured output
            format = {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The main response content"
                    },
                    "key_points": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of key points from the response"
                    }
                }
            }
        
        main_response = generate_text(
            main_model, prompt, max_length, temperature, top_k, top_p, num_sequences, None, context, format
        )
        
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": main_response})

        audio_output = None
        if generate_voice:
            audio_output = text_to_speech(main_response, voice_name)

        return (
            chat_history_to_string(),
            "",
            chat_history_to_string(),
            audio_output,
            gr.update(value=""),
            gr.update(value=""),
        )

def chat_history_to_string() -> str:
    """Format chat history as HTML for display with improved formatting"""
    chat_html = '<div style="display: flex; flex-direction: column; gap: 15px; font-size: 16px; width: 100%;">'
    
    for msg in chat_history:
        # Determine style based on role
        if msg['role'] == 'user':
            style = "align-self: flex-end; background-color: #1982FC; color: white;"
            role_align = "text-align: right;"
        else:
            style = "align-self: flex-start; background-color: #248a3e; color: white;"
            role_align = ""
        
        # Process content to preserve formatting
        content = msg["content"]
        
        # Preserve paragraph breaks by replacing newlines with <br> tags
        content = content.replace('\n\n', '<br><br>')
        content = content.replace('\n', '<br>')
        
        # Format code blocks - detect markdown code blocks
        content = re.sub(
            r'```(.*?)```', 
            r'<pre style="background-color: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 5px; overflow-x: auto; font-family: monospace;">\1</pre>', 
            content, 
            flags=re.DOTALL
        )
        
        # Add the message to chat HTML
        chat_html += f'''
        <div style="{style} max-width: 80%; padding: 12px 18px; border-radius: 20px; 
                    position: relative; word-wrap: break-word; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 1.1em; white-space: pre-wrap; line-height: 1.5;">{content}</div>
            <div style="font-size: 0.8em; opacity: 0.7; {role_align} margin-top: 5px;">{msg['role'].capitalize()}</div>
        </div>
        '''
    
    chat_html += '</div>'
    return chat_html

def cycle_markdown(direction: str) -> str:
    """Navigate through markdown history"""
    global current_markdown_index
    if direction == "next" and current_markdown_index < len(markdown_history) - 1:
        current_markdown_index += 1
    elif direction == "prev" and current_markdown_index > 0:
        current_markdown_index -= 1
    return markdown_history[current_markdown_index] if markdown_history else ""

def record_audio(progress=gr.Progress()) -> str:
    """Record audio from microphone and convert to text"""
    recognizer = sr.Recognizer()
    recording = True
    audio_data = []

    def record():
        nonlocal recording, audio_data
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            while recording:
                try:
                    audio_chunk = recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    audio_data.append(audio_chunk)
                except sr.WaitTimeoutError:
                    pass

    thread = threading.Thread(target=record)
    thread.start()

    # Show recording progress for 10 seconds max
    for _ in progress.tqdm(range(100)):
        if not recording:
            break
        time.sleep(0.1)

    recording = False
    thread.join()

    if not audio_data:
        return "No audio recorded."

    full_audio = sr.AudioData(
        b"".join(chunk.get_raw_data() for chunk in audio_data),
        audio_data[0].sample_rate,
        audio_data[0].sample_width,
    )

    try:
        text = recognizer.recognize_google(full_audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio. Please try again."
    except sr.RequestError as e:
        return f"Could not request results; {e}. Please check your internet connection."

def create_or_switch_session(new_session_name=None):
    """Create a new session or switch to an existing one"""
    global current_session, chat_history, markdown_history, sessions
    
    # Save current session if it exists
    if current_session:
        sessions[current_session] = (chat_history.copy(), markdown_history.copy())
    
    # Create new session if name provided
    if new_session_name:
        session_name = new_session_name
    else:
        session_name = f"Session {len(sessions) + 1}"
    
    # Set as current and clear history
    current_session = session_name
    chat_history = []
    markdown_history = []
    
    # Return updated session info
    session_choices = list(sessions.keys()) + [current_session]
    if current_session not in session_choices:
        session_choices.append(current_session)
    
    return gr.update(choices=session_choices, value=current_session), "", ""

def load_session(session_name: str) -> Tuple[str, str]:
    """Load a saved session"""
    global current_session, chat_history, markdown_history
    if session_name in sessions:
        current_session = session_name
        chat_history, markdown_history = sessions[session_name]
    else:
        chat_history = []
        markdown_history = []
    return chat_history_to_string(), markdown_history[current_markdown_index] if markdown_history else ""

def delete_session(session_name: str) -> Tuple[gr.update, str, str]:
    """Delete a session"""
    global current_session, chat_history, markdown_history
    if session_name in sessions:
        del sessions[session_name]
        if current_session == session_name:
            current_session = None
            chat_history = []
            markdown_history = []
    session_choices = list(sessions.keys())
    return gr.update(choices=session_choices), "", ""

def download_code(code: str):
    """Save code to a file for download"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        f.write(code)
    return f.name

def download_model_with_progress(model_name: str, progress=gr.Progress()) -> str:
    """Download a model from Ollama with progress tracking"""
    try:
        last_status = ""
        total = 0
        completed = 0
        
        for progress_data in client.pull(model_name, stream=True):
            status = progress_data.get('status', '')
            
            # Handle different progress information formats
            if 'total' in progress_data and progress_data['total'] > 0:
                total = progress_data['total']
                
            if 'completed' in progress_data and progress_data['completed'] > 0:
                completed = progress_data['completed']
                
            # Update progress bar when we have meaningful data
            if total > 0 and completed > 0:
                progress_percentage = min(completed / total, 1.0)
                progress(progress_percentage, desc=f"Downloading {model_name}")
            elif status and status != last_status:
                # When we don't have numeric progress, use status updates
                last_status = status
                progress(0.5, desc=f"Status: {status}")
                
        return f"Model '{model_name}' downloaded successfully."
    except Exception as e:
        return f"Error downloading model: {str(e)}"

def delete_model(model_name: str) -> str:
    """Delete a model from Ollama"""
    try:
        client.delete(model_name)
        return f"Model '{model_name}' deleted successfully."
    except Exception as e:
        return f"Error deleting model: {str(e)}"

def load_modelfile(model_name: str) -> str:
    """Load modelfile content for a model"""
    try:
        model_info = client.show(model_name)
        return model_info.get('modelfile', 'No modelfile content found.')
    except Exception as e:
        return f"Error loading modelfile: {str(e)}"

def save_modelfile(model_name: str, modelfile_content: str) -> str:
    """Save modelfile content and create a model"""
    try:
        modelfile_path = f"{model_name}.modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        subprocess.run(['ollama', 'create', model_name, '-f', modelfile_path], check=True)
        os.remove(modelfile_path)
        return f"Modelfile for '{model_name}' saved and applied successfully."
    except Exception as e:
        return f"Error saving modelfile: {str(e)}"

def update_model_list():
    """Update all model dropdown lists"""
    models = get_available_models()
    return (gr.update(choices=models),) * 5  # Update all model dropdowns

def execute_code(code: str) -> str:
    """Execute Python code in a safe environment"""
    try:
        # Prepare a safe environment
        safe_globals = {"__builtins__": {}}
        exec(code, safe_globals)
        return "Code executed successfully."
    except Exception as e:
        return f"Error executing code: {str(e)}"

def lint_code(code: str) -> str:
    """Lint Python code for syntax errors"""
    try:
        # Simple linting: check for syntax errors
        compile(code, '<string>', 'exec')
        return "No syntax errors found."
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return f"An error occurred during linting: {str(e)}"

def build_rag_index(documents, model_name="llama3"):
    """Build a RAG (Retrieval Augmented Generation) index from documents"""
    try:
        if not documents:
            return "No documents provided for indexing."
            
        # Extract text from documents
        texts = []
        for doc in documents:
            text = extract_text_from_document(doc)
            if text:
                texts.append(text)
                
        if not texts:
            return "Could not extract text from the provided documents."
            
        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.create_documents(texts)
        
        # Create vector store directory with model name to avoid dimension conflicts
        persist_directory = f"./chroma_db_{model_name}"
        
        # Create embeddings and vectorstore
        embeddings = OllamaEmbeddings(model=model_name)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Save the model name used for embeddings for later use
        with open(f"{persist_directory}/model_info.txt", "w") as f:
            f.write(model_name)
        
        return f"RAG index built successfully with {len(chunks)} chunks using the {model_name} model."
    except Exception as e:
        logger.error(f"Error building RAG index: {e}")
        return f"Error building RAG index: {str(e)}"

def query_rag(query, model_name):
    """Query the RAG system with a user question"""
    try:
        # Use the same model directory that was used to create the index
        persist_directory = f"./chroma_db_{model_name}"
        
        # Check if vectorstore exists
        if not os.path.exists(persist_directory):
            return f"No RAG index found for model {model_name}. Please build the index first using this model."
        
        # Get the model name used for building the index
        embedding_model = model_name
        if os.path.exists(f"{persist_directory}/model_info.txt"):
            with open(f"{persist_directory}/model_info.txt", "r") as f:
                embedding_model = f.read().strip()
        
        # Load embeddings using the same model that was used to create the index
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Load vectorstore
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response with context
        prompt = f"""
You are a helpful assistant answering questions based on provided context information.

CONTEXT:
{context}

USER QUESTION:
{query}

Please answer the question based only on the context provided. If the answer cannot be found in the context, say "I don't have enough information to answer that question based on the provided documents."
"""
        
        response = client.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": 0.7,
            }
        )
        
        return response["response"]
    except Exception as e:
        logger.error(f"Error querying RAG: {e}")
        return f"Error querying RAG system: {str(e)}"

def clear_chat():
    """Clear the chat history"""
    global chat_history
    chat_history = []
    return chat_history_to_string(), ""

def sync_settings_to_quick(settings_temp, settings_maxlen, settings_topk, settings_topp, settings_numseq):
    """Sync settings from Settings tab to Quick Settings"""
    return [
        gr.update(value=settings_temp),
        gr.update(value=settings_maxlen),
        gr.update(value=settings_topk),
        gr.update(value=settings_topp),
        gr.update(value=settings_numseq)
    ]

def sync_quick_to_settings(quick_temp, quick_maxlen, quick_topk, quick_topp, quick_numseq):
    """Sync settings from Quick Settings to Settings tab"""
    return [
        gr.update(value=quick_temp),
        gr.update(value=quick_maxlen),
        gr.update(value=quick_topk),
        gr.update(value=quick_topp),
        gr.update(value=quick_numseq)
    ]

# File upload functions
def update_file_preview(files):
    """Update file preview with list of uploaded document names"""
    if not files or len(files) == 0:
        return "<div id='file-list'><div style='opacity:0.7;text-align:center;padding:10px;'>No files uploaded</div></div>"

    html = "<div id='file-list' style='max-height:150px;overflow-y:auto;'>"
    for file in files:
        file_name = os.path.basename(file.name) if hasattr(file, 'name') else "File"
        file_ext = os.path.splitext(file_name)[1].lower() if '.' in file_name else ""
        
        # Choose icon based on file extension
        icon = "📄"
        if file_ext in ['.pdf']:
            icon = "📑"
        elif file_ext in ['.docx', '.doc']:
            icon = "📝"
        elif file_ext in ['.txt']:
            icon = "📃"
        
        html += f"<div style='padding:5px;margin:3px 0;background:#f0f0f0;border-radius:4px;display:flex;align-items:center;'>"
        html += f"<span style='margin-right:8px;'>{icon}</span> {file_name}</div>"
    html += "</div>"
    return html

def enhanced_clear_image_on_document_upload(document_files):
    """Clear image upload when documents are uploaded and update status/preview"""
    if document_files and len(document_files) > 0:
        file_count = len(document_files)
        status_msg = f"✅ {file_count} document(s) uploaded"
        file_preview_html = update_file_preview(document_files)
        return gr.update(value=None), status_msg, file_preview_html
    return gr.update(), "No documents uploaded", "<div id='file-list'><div style='opacity:0.7;text-align:center;padding:10px;'>No files uploaded</div></div>"

def enhanced_clear_document_on_image_upload(image_file):
    """Clear document upload when an image is uploaded and update status/preview"""
    if image_file is not None:
        image_name = os.path.basename(image_file) if isinstance(image_file, str) else "Image"
        status_msg = f"✅ Image uploaded"
        return gr.update(value=[]), status_msg, "<div id='file-list'><div style='opacity:0.7;text-align:center;padding:10px;'>No files uploaded</div></div>"
    return gr.update(), "No image uploaded", gr.update()

# Gradio interface setup
def create_interface():
    """Create the main Gradio interface with improved UI"""
    with gr.Blocks(
        title="Geeky Ollama WebUI",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        #chat-container { 
            display: flex; 
            flex-direction: column; 
            gap: 10px; 
            height: 500px; 
            overflow-y: auto; 
            padding: 10px; 
            border: 1px solid #ddd; 
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .chat-message { 
            padding: 12px 18px; 
            margin: 5px 0; 
            border-radius: 15px; 
            max-width: 80%; 
            position: relative;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 16px;
            line-height: 1.5;
        }
        .user-message { 
            background-color: #007bff; 
            color: #ecf0f1; 
            align-self: flex-end;
            border-bottom-right-radius: 0;
        }
        .assistant-message { 
            background-color: #000000; 
            color: #ecf0f1; 
            align-self: flex-start;
            border-bottom-left-radius: 0;
        }
        .chat-message small {
            display: block;
            margin-top: 8px;
            opacity: 0.8;
            font-size: 0.85em;
        }
        .user-message small, .assistant-message small { 
            color: #bdc3c7;
        }
        .code-area { 
            font-family: 'Courier New', monospace; 
            background-color: #27282b; 
            border: 1px solid #ddd;
            font-size: 14px;
        }
        .tab-content { padding: 15px; border: 1px solid #ddd; border-top: none; }
        .gr-button { transition: all 0.3s ease; }
        .gr-button:hover { transform: translateY(-2px); box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        .markdown-text p { margin-bottom: 10px; }
        .markdown-text h1, .markdown-text h2, .markdown-text h3 { margin-top: 20px; }
        .settings-block { 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            padding: 10px; 
            margin-bottom: 15px; 
            background-color: #f9f9f9;
        }
        .settings-label {
            font-weight: bold;
            margin-bottom: 8px;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            font-size: 0.9em;
            color: #666;
        }
        
        /* Upload section styling */
        #upload-container {
            padding: 15px;
            border-radius: 8px;
            background-color: #f8f8fa;
            margin-bottom: 10px;
        }
        
        #image-upload, #document-upload {
            border: 1px dashed #ccc;
            border-radius: 8px;
            padding: 10px;
            background-color: #fafafa;
            min-height: 150px;
        }
        
        #image-status, #document-status {
            margin-top: 5px;
            font-size: 14px;
            color: #555;
            text-align: center;
            padding: 5px;
            border-radius: 4px;
            background-color: #f0f0f0;
        }
        
        #file-preview-container {
            margin-top: 10px;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #eee;
            background-color: #f5f5f5;
        }
        
        #file-list {
            max-height: 150px;
            overflow-y: auto;
            padding: 5px;
        }
        
        #file-list > div {
            padding: 5px;
            margin: 3px 0;
            background: #f0f0f0;
            border-radius: 4px;
            display: flex;
            align-items: center;
        }
        
        @media (max-width: 768px) {
            .gr-form { flex-direction: column; }
        }
        """
    ) as interface:
        # Header
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    # 🤖 Geeky Ollama WebUI
                    An intuitive interface for interacting with Ollama AI models
                    """
                )
            with gr.Column(scale=1, min_width=120):
                refresh_button = gr.Button("🔄 Refresh Models")

        # Main layout with tabs
        with gr.Tabs(selected=0) as main_tabs:
            # Chat Tab - Simplified primary interface
            with gr.TabItem("💬 Chat", id="chat-tab"):
                # Define audio output here, before it's referenced in event handlers
                audio_output = gr.Audio(label="Voice Output", visible=False, autoplay=True)
                
                with gr.Row():
                    # Main chat area
                    with gr.Column(scale=3):
                        chat_display = gr.HTML(label="Conversation", elem_id="chat-display")
                        
                        with gr.Row():
                            input_text = gr.Textbox(
                                lines=3,
                                label="Your Message",
                                placeholder="Type your message here...",
                                elem_id="input-text"
                            )
                            mic_button = gr.Button("🎤", elem_id="mic-button")
                        
                        with gr.Row():
                            generate_button = gr.Button("Send 📤", elem_id="generate-button", variant="primary")
                            clear_button = gr.Button("Clear Chat 🧹", elem_id="clear-button")
                    
                    # Sidebar with essential controls
                    with gr.Column(scale=1):
                        # Get models once to avoid duplicate calls
                        available_models = get_available_models()
                        main_model_dropdown = gr.Dropdown(
                            choices=available_models,
                            label="Select Model",
                            value=available_models[0] if available_models else None,
                        )
                        
                        # Mode selection
                        mode_radio = gr.Radio(
                            ["Chat", "Coding", "Image Analysis", "Function Calling", "Structured Output", "Document QA"],
                            label="Mode",
                            value="Chat",
                            elem_id="mode-radio"
                        )
                        
                        # Quick settings
                        with gr.Accordion("Quick Settings", open=False):
                            temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                            max_length = gr.Slider(50, 4200, value=500, step=50, label="Max Length")
                            top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k")
                            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                            num_sequences = gr.Slider(1, 5, value=1, step=1, label="Sequences")
                            
                            with gr.Row():
                                json_output_checkbox = gr.Checkbox(label="JSON Output", value=False)
                                function_calling_checkbox = gr.Checkbox(label="Function Calling", value=False)
                            
                            with gr.Row():
                                generate_voice_checkbox = gr.Checkbox(label="Generate Voice", value=False)
                                voice_dropdown = gr.Dropdown(
                                    choices=get_available_voices(),
                                    label="Voice",
                                    value=get_available_voices()[0] if get_available_voices() else None,
                                    visible=False
                                )
                        
                        # Session controls
                        with gr.Accordion("Session Management", open=False):
                            with gr.Row():
                                new_session_button = gr.Button("New Session 🆕", size="sm")
                                session_name_input = gr.Textbox(label="Session Name (optional)")
                            
                            with gr.Row():
                                session_dropdown = gr.Dropdown(choices=[], label="Select Session")
                                load_session_button = gr.Button("Load", size="sm")
                                delete_session_button = gr.Button("Delete", size="sm")
                        
                        # IMPROVED UPLOADS SECTION
                        with gr.Accordion("Uploads", open=False):
                            # Document Upload (FIRST - above the image upload)
                            gr.Markdown("### 📄 Document Upload")
                            document_input = gr.File(
                                label="Upload Documents for Context", 
                                file_count="multiple",
                                elem_id="document-upload",
                                file_types=["pdf", "txt", "docx"]
                            )
                            document_status = gr.Markdown("No documents uploaded", elem_id="document-status")
                            
                            # Document file preview - shows names of uploaded files
                            file_preview = gr.HTML(
                                value="<div id='file-list'><div style='opacity:0.7;text-align:center;padding:10px;'>No files uploaded</div></div>", 
                                label="Uploaded Files"
                            )
                            
                            # Divider
                            gr.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #555;'>")
                            
                            # Image Upload (SECOND - below the document upload)
                            gr.Markdown("### 🖼️ Image Upload")
                            image_input = gr.Image(
                                type="filepath", 
                                label="Upload Image for Analysis",
                                elem_id="image-upload",
                                sources=["upload", "clipboard", "webcam"]
                            )
                            image_status = gr.Markdown("No image uploaded", elem_id="image-status")

            # Code Generation tab - Optimized
            with gr.TabItem("💻 Code", id="code-tab"):
                with gr.Row():
                    # Code area
                    with gr.Column(scale=3):
                        code_output = gr.Code(
                            label="Code Output", 
                            language="python", 
                            elem_classes="code-area",
                            lines=20
                        )
                        
                        with gr.Row():
                            prev_button = gr.Button("◀ Previous", size="sm")
                            next_button = gr.Button("Next ▶", size="sm")
                            download_code_button = gr.Button("💾 Download", size="sm")
                            copy_button = gr.Button("📋 Copy", size="sm")
                        
                        with gr.Row():
                            execute_button = gr.Button("▶️ Execute", size="sm")
                            lint_button = gr.Button("🔍 Lint", size="sm")
                        
                        code_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Code settings
                    with gr.Column(scale=1):
                        coding_model_dropdown = gr.Dropdown(
                            choices=available_models,
                            label="Coding Model",
                            value=available_models[0] if available_models else None,
                        )
                        
                        #"Code Generation Settings" accordion
                        with gr.Accordion("Code Generation Settings", open=False):
                            code_temperature = gr.Slider(
                                0.1, 1.0, value=0.1, step=0.1, 
                                label="Temperature",
                                elem_id="code-temp-slider"
                            )
                            code_max_length = gr.Slider(
                                50, 8000, value=2000, step=50, 
                                label="Max Length",
                                elem_id="code-maxlen-slider"
                            )

            # Document QA tab - Fixed RAG implementation
            with gr.TabItem("📚 Document QA", id="docqa-tab"):
                with gr.Row():
                    with gr.Column(scale=3):
                        rag_query = gr.Textbox(
                            label="Ask a question about your documents",
                            placeholder="What information can you find about...?",
                            lines=2
                        )
                        
                        rag_response = gr.Textbox(label="Answer", lines=8)
                        
                        with gr.Row():
                            rag_query_button = gr.Button("Query Documents", size="sm", variant="primary")
                            clear_rag_button = gr.Button("Clear", size="sm")
                    
                    with gr.Column(scale=1):
                        rag_model = gr.Dropdown(
                            choices=available_models,
                            label="Select Model",
                            value=available_models[0] if available_models else None
                        )
                        
                        rag_documents = gr.File(
                            label="Upload Documents", 
                            file_count="multiple"
                        )
                        
                        build_index_button = gr.Button("Build Knowledge Base", size="sm", variant="primary")
                        rag_status = gr.Textbox(label="Status", interactive=False)

            # Model Management tab - Enhanced
            with gr.TabItem("🔧 Models", id="model-tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group(elem_classes="settings-block"):
                            gr.Markdown("### Download Models", elem_classes="settings-label")
                            model_name_input = gr.Textbox(label="Model Name (e.g., llama3, llama3:8b)")
                            download_model_button = gr.Button("Download Model", size="sm", variant="primary")
                        
                        with gr.Group(elem_classes="settings-block"):
                            gr.Markdown("### Manage Existing Models", elem_classes="settings-label")
                            with gr.Row():
                                delete_model_dropdown = gr.Dropdown(
                                    choices=available_models, 
                                    label="Select Model"
                                )
                                delete_model_button = gr.Button("Delete Model", size="sm")
                            
                            with gr.Row():
                                load_model_dropdown = gr.Dropdown(
                                    choices=available_models, 
                                    label="Select Model"
                                )
                                load_modelfile_button = gr.Button("Load Modelfile", size="sm")
                    
                    with gr.Column(scale=2):
                        modelfile_input = gr.TextArea(
                            lines=12, 
                            label="Modelfile Content",
                            placeholder="FROM llama3\nSYSTEM \"You are a helpful assistant.\""
                        )
                        save_modelfile_button = gr.Button("Save Modelfile", size="sm")
                        model_management_output = gr.Textbox(label="Output", lines=3)

            # Settings tab - Better organized
            with gr.TabItem("⚙️ Settings", id="settings-tab"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes="settings-block"):
                            gr.Markdown("### Generation Parameters", elem_classes="settings-label")
                            with gr.Row():
                                with gr.Column():
                                    settings_max_length = gr.Slider(50, 4200, value=500, step=50, label="Max Length")
                                    settings_temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                                
                                with gr.Column():
                                    settings_top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k")
                                    settings_top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                            
                            settings_num_sequences = gr.Slider(1, 5, value=1, step=1, label="Number of Sequences")
                        
                        with gr.Group(elem_classes="settings-block"):
                            gr.Markdown("### Voice Settings", elem_classes="settings-label")
                            settings_voice_dropdown = gr.Dropdown(
                                choices=get_available_voices(),
                                label="TTS Voice",
                                value=get_available_voices()[0] if get_available_voices() else None,
                            )
                    
                    with gr.Column():
                        with gr.Group(elem_classes="settings-block"):
                            gr.Markdown("### Advanced Options", elem_classes="settings-label")
                            settings_json_output = gr.Checkbox(label="Enable Structured JSON Output", value=False)
                            settings_function_calling = gr.Checkbox(label="Enable Function Calling", value=False)
                        
                        with gr.Group(elem_classes="settings-block"):
                            gr.Markdown("### About", elem_classes="settings-label")
                            gr.Markdown("""
                            **Geeky Ollama WebUI** provides an intuitive interface for working with Ollama models.
                            
                            - Chat with various models
                            - Generate and execute code
                            - Analyze images using multimodal models
                            - Ask questions about your documents
                            - Manage your Ollama models
                            
                            [Ollama Website](https://ollama.ai) | [GitHub](https://github.com/ollama/ollama)
                            """, elem_classes="markdown-text")

        # Footer
        gr.Markdown("""
        <div class="footer">
        Geeky Ollama WebUI - Powered by Gradio & Ollama
        </div>
        """)

        # Event handlers
        # Chat functionality
        generate_button.click(
            generate_with_context,
            inputs=[
                main_model_dropdown,
                coding_model_dropdown,
                input_text,
                max_length,
                temperature,
                top_k,
                top_p,
                num_sequences,
                image_input,
                document_input,
                mode_radio,
                voice_dropdown,
                generate_voice_checkbox,
                json_output_checkbox,
                function_calling_checkbox,
            ],
            outputs=[chat_display, code_output, chat_display, audio_output, code_output, input_text],
        )

        # Code navigation
        prev_button.click(lambda: cycle_markdown("prev"), outputs=[code_output])
        next_button.click(lambda: cycle_markdown("next"), outputs=[code_output])

        # Voice checkbox visibility toggle
        generate_voice_checkbox.change(
            lambda x: gr.update(visible=x), 
            inputs=[generate_voice_checkbox], 
            outputs=[voice_dropdown]
        )

        # Microphone button
        mic_button.click(record_audio, outputs=[input_text])

        # Session management
        new_session_button.click(
            lambda name: create_or_switch_session(name if name else None), 
            inputs=[session_name_input], 
            outputs=[session_dropdown, chat_display, code_output]
        )
        
        load_session_button.click(
            load_session, 
            inputs=[session_dropdown], 
            outputs=[chat_display, code_output]
        )
        
        delete_session_button.click(
            delete_session, 
            inputs=[session_dropdown], 
            outputs=[session_dropdown, chat_display, code_output]
        )

        # Voice settings
        generate_voice_checkbox.change(
            lambda x: gr.update(visible=x), 
            inputs=[generate_voice_checkbox], 
            outputs=[audio_output]
        )

        # Code actions
        download_code_button.click(download_code, inputs=[code_output], outputs=[gr.File()])
        copy_button.click(lambda x: x, inputs=[code_output], outputs=[gr.Textbox(visible=False)])
        execute_button.click(execute_code, inputs=[code_output], outputs=[code_status])
        lint_button.click(lint_code, inputs=[code_output], outputs=[code_status])

        # File upload event handlers - enhanced versions with file preview
        document_input.change(
            enhanced_clear_image_on_document_upload,
            inputs=[document_input],
            outputs=[image_input, image_status, file_preview]
        )

        image_input.change(
            enhanced_clear_document_on_image_upload,
            inputs=[image_input],
            outputs=[document_input, document_status, file_preview]
        )

        # Settings sync event handlers
        settings_temperature.change(
            sync_settings_to_quick,
            inputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences],
            outputs=[temperature, max_length, top_k, top_p, num_sequences]
        )
        
        settings_max_length.change(
            sync_settings_to_quick,
            inputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences],
            outputs=[temperature, max_length, top_k, top_p, num_sequences]
        )
        
        settings_top_k.change(
            sync_settings_to_quick,
            inputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences],
            outputs=[temperature, max_length, top_k, top_p, num_sequences]
        )
        
        settings_top_p.change(
            sync_settings_to_quick,
            inputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences],
            outputs=[temperature, max_length, top_k, top_p, num_sequences]
        )
        
        settings_num_sequences.change(
            sync_settings_to_quick,
            inputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences],
            outputs=[temperature, max_length, top_k, top_p, num_sequences]
        )
        
        # Quick settings to main settings sync
        temperature.change(
            sync_quick_to_settings,
            inputs=[temperature, max_length, top_k, top_p, num_sequences],
            outputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences]
        )
        
        max_length.change(
            sync_quick_to_settings,
            inputs=[temperature, max_length, top_k, top_p, num_sequences],
            outputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences]
        )
        
        top_k.change(
            sync_quick_to_settings,
            inputs=[temperature, max_length, top_k, top_p, num_sequences],
            outputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences]
        )
        
        top_p.change(
            sync_quick_to_settings,
            inputs=[temperature, max_length, top_k, top_p, num_sequences],
            outputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences]
        )
        
        num_sequences.change(
            sync_quick_to_settings,
            inputs=[temperature, max_length, top_k, top_p, num_sequences],
            outputs=[settings_temperature, settings_max_length, settings_top_k, settings_top_p, settings_num_sequences]
        )

        # Model management
        download_model_button.click(
            download_model_with_progress, 
            inputs=[model_name_input], 
            outputs=[model_management_output]
        )
        
        delete_model_button.click(
            delete_model, 
            inputs=[delete_model_dropdown], 
            outputs=[model_management_output]
        )
        
        load_modelfile_button.click(
            load_modelfile, 
            inputs=[load_model_dropdown], 
            outputs=[modelfile_input]
        )
        
        save_modelfile_button.click(
            save_modelfile, 
            inputs=[model_name_input, modelfile_input], 
            outputs=[model_management_output]
        )

        # Refresh models
        refresh_button.click(
            update_model_list,
            outputs=[
                main_model_dropdown,
                coding_model_dropdown,
                rag_model,
                delete_model_dropdown,
                load_model_dropdown,
            ],
        )

        # Model operations model list updates
        for button in [download_model_button, delete_model_button, save_modelfile_button]:
            button.click(
                update_model_list,
                outputs=[
                    main_model_dropdown,
                    coding_model_dropdown,
                    rag_model,
                    delete_model_dropdown,
                    load_model_dropdown,
                ],
            )

        # RAG operations
        build_index_button.click(
            lambda docs, model: build_rag_index(docs, model),
            inputs=[rag_documents, rag_model],
            outputs=[rag_status]
        )
        
        rag_query_button.click(
            query_rag,
            inputs=[rag_query, rag_model],
            outputs=[rag_response]
        )
        
        clear_rag_button.click(
            lambda: "",
            outputs=[rag_response]
        )

        # Clear chat
        clear_button.click(
            clear_chat,
            outputs=[chat_display, input_text]
        )

    return interface

def markdown_history_to_string() -> str:
    """Format markdown history"""
    return markdown_history[current_markdown_index] if markdown_history else ""

if __name__ == "__main__":
    try:
        interface = create_interface()
        interface.launch(share=False, server_name="127.0.0.1")
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")
        print(f"An error occurred while launching the interface: {e}")
        traceback.print_exc()