import gradio as gr
import ollama
import json
import base64
import os
import subprocess
from PyPDF2 import PdfReader
from docx import Document
import pyttsx3
import concurrent.futures
import logging
import speech_recognition as sr
from typing import List, Dict, Optional, Tuple
from functools import partial
import re
import ast
import jedi
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
import threading
import time
import pkgutil
import importlib
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_API_URL = "http://localhost:11434"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pubmed_vectors"

# Create an Ollama client
client = ollama.Client(host=OLLAMA_API_URL)

# Create a Qdrant client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Global variables
chat_history: List[Dict[str, str]] = []
markdown_history: List[str] = []
current_markdown_index: int = 0
sessions: Dict[str, Tuple[List[Dict[str, str]], List[str]]] = {}
current_session: Optional[str] = None
code_versions: List[str] = []

def get_available_models() -> List[str]:
    try:
        models = client.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return ["No models found"]

def generate_text(model: str, prompt: str, max_length: int, temperature: float, top_k: int, top_p: float,
                  num_sequences: int, image: Optional[str] = None, context: Optional[str] = None) -> str:
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

    if image:
        with open(image, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        options['images'] = [img_base64]

    try:
        responses = []
        for _ in range(num_sequences):
            response = client.generate(**options)
            generated_text = response['response'].strip()
            responses.append(generated_text)

        return "\n\n--- New Sequence ---\n\n".join(responses)
    except Exception as e:
        logger.error(f"Error in generate_text: {e}")
        return f"An error occurred: {str(e)}"

def extract_text_from_document(file) -> Optional[str]:
    if file is None:
        return None

    try:
        if file.name.endswith('.pdf'):
            reader = PdfReader(file.name)
            return "\n".join(page.extract_text() for page in reader.pages)
        elif file.name.endswith('.docx'):
            doc = Document(file.name)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif file.name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return "Unsupported file format"
    except Exception as e:
        logger.error(f"Error extracting text from document: {e}")
        return f"Error processing document: {str(e)}"

def get_available_voices() -> List[str]:
    engine = pyttsx3.init()
    return [voice.name for voice in engine.getProperty('voices')]

def text_to_speech(text: str, voice_name: str) -> Optional[str]:
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

def get_embedding(text: str) -> List[float]:
    # Use OllamaEmbeddings for consistency with the rest of the system
    embeddings = OllamaEmbeddings(model="llama2")
    return embeddings.embed_query(text)

def retrieve_context(query: str) -> str:
    try:
        query_vector = get_embedding(query)
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        contexts = [hit.payload.get('text', '') for hit in search_result]
        return "\n".join(contexts)
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""

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
):
    global chat_history, markdown_history, current_markdown_index

    # Retrieve context using RAG
    context = retrieve_context(prompt)

    if mode == "Coding":
        explanation_prompt = f"Context: {context}\n\nUser Request: {prompt}\n\nProvide an explanation of the proposed changes and how to use them. Do not include any code in this response."
        explanation = generate_text(main_model, explanation_prompt, max_length, temperature, top_k, top_p, 1)

        coding_prompt = f"Generate Python code based on the following request. Only output code with proper comments. Do not include any explanations outside of code comments.\n\nContext: {context}\n\nUser Request: {prompt}\n\nMain Model Explanation: {explanation}"
        code_response = generate_text(coding_model, coding_prompt, max_length, temperature, top_k, top_p, 1)

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
    else:
        main_response = generate_text(
            main_model, prompt, max_length, temperature, top_k, top_p, num_sequences, image, context
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
    chat_html = '<div id="chat-container">'
    for msg in chat_history:
        class_name = "user-message" if msg['role'] == 'user' else "assistant-message"
        chat_html += f'<div class="chat-message {class_name}">'
        
        # Format the content
        formatted_content = msg["content"].replace('\n', '<br>')
        formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_content)
        formatted_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_content)
        formatted_content = re.sub(r'`(.*?)`', r'<code>\1</code>', formatted_content)
        
        # Add special formatting for lists
        formatted_content = re.sub(r'(?m)^(\d+\.|\-)\s', r'<br>• ', formatted_content)
        
        chat_html += f'<p>{formatted_content}</p>'
        chat_html += f'<small>{msg["role"].capitalize()}</small>'
        chat_html += '</div>'
    chat_html += '</div>'
    return chat_html

def cycle_markdown(direction: str) -> str:
    global current_markdown_index
    if direction == "next" and current_markdown_index < len(markdown_history) - 1:
        current_markdown_index += 1
    elif direction == "prev" and current_markdown_index > 0:
        current_markdown_index -= 1
    return markdown_history[current_markdown_index]

def record_audio(progress=gr.Progress()) -> str:
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

def new_session() -> Tuple[gr.update, str, str]:
    global current_session, chat_history, markdown_history
    if current_session:
        sessions[current_session] = (chat_history, markdown_history)
    current_session = f"Session {len(sessions) + 1}"
    chat_history = []
    markdown_history = []
    session_choices = list(sessions.keys()) + [current_session]
    return gr.update(choices=session_choices, value=current_session), "", ""

def load_session(session_name: str) -> Tuple[str, str]:
    global current_session, chat_history, markdown_history
    if session_name in sessions:
        current_session = session_name
        chat_history, markdown_history = sessions[session_name]
    else:
        chat_history = []
        markdown_history = []
    return chat_history_to_string(), markdown_history_to_string()

def delete_session(session_name: str) -> Tuple[gr.update, str, str]:
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
    filename = "code_output.py"
    with open(filename, "w") as f:
        f.write(code)
    return filename

def download_model(model_name: str) -> str:
    try:
        client.pull_model(model_name)
        return f"Model '{model_name}' downloaded successfully."
    except Exception as e:
        return f"Error downloading model: {str(e)}"

def delete_model(model_name: str) -> str:
    try:
        client.delete_model(model_name)
        return f"Model '{model_name}' deleted successfully."
    except Exception as e:
        return f"Error deleting model: {str(e)}"

def load_modelfile(model_name: str) -> str:
    try:
        model_info = client.show_model(model_name)
        return model_info.get('modelfile', 'No modelfile content found.')
    except Exception as e:
        return f"Error loading modelfile: {str(e)}"

def save_modelfile(model_name: str, modelfile_content: str) -> str:
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
    models = get_available_models()
    return (gr.update(choices=models),) * 4  # Update four dropdowns

def execute_code(code: str) -> str:
    try:
        # Prepare a safe environment
        safe_globals = {"__builtins__": {}}
        exec(code, safe_globals)
        return "Code executed successfully."
    except Exception as e:
        return f"Error executing code: {str(e)}"

def lint_code(code: str) -> str:
    try:
        # Simple linting: check for syntax errors
        compile(code, '<string>', 'exec')
        return "No syntax errors found."
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return f"An error occurred during linting: {str(e)}"

def explore_modules() -> List[str]:
    return [name for _, name, _ in pkgutil.iter_modules()]

def import_module(module_name: str) -> str:
    try:
        importlib.import_module(module_name)
        return f"Module '{module_name}' imported successfully."
    except ImportError as e:
        return f"Error importing module '{module_name}': {str(e)}"

def process_document(file_path: str, question: str) -> str:
    # Extract text from the document
    text = extract_text_from_document(file_path)
    
    # Use RAG to get context
    context = retrieve_context(question)
    
    # Generate response using the main model
    response = generate_text("main_model", f"Context: {context}\n\nDocument: {text}\n\nQuestion: {question}", 500, 0.7, 40, 0.9, 1)
    
    return response

# Gradio interface setup
def create_interface():
    with gr.Blocks(
        title="PubMed Faster RAG!",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        #chat-container { 
            display: flex; 
            flex-direction: column; 
            gap: 10px; 
            height: 400px; 
            overflow-y: auto; 
            padding: 10px; 
            border: 1px solid #ddd; 
            border-radius: 5px;
            background-color: #4C4E52;
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
            background-color: #34495e; 
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
        @media (max-width: 768px) {
            .gr-form { flex-direction: column; }
        }
        """
    ) as iface:
        gr.Markdown("# 🚀 PubMed Faster RAG!")

        with gr.Row():
            with gr.Column(scale=2):
                chat_display = gr.HTML(label="Chat History", elem_id="chat-display")
                with gr.Row():
                    input_text = gr.Textbox(
                        lines=3,
                        label="Your Message",
                        placeholder="Ask anything...",
                        elem_id="input-text"
                    )
                    mic_button = gr.Button("🎤", elem_id="mic-button")
                with gr.Row():
                    generate_button = gr.Button("Send 📤", elem_id="generate-button", variant="primary")
                    clear_button = gr.Button("Clear Chat 🧹", elem_id="clear-button")

            with gr.Column(scale=1):
                main_model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Main Model",
                    value=get_available_models()[0] if get_available_models() else None,
                )
                mode_radio = gr.Radio(
                    ["Chat", "Coding", "Document QA"],
                    label="Mode",
                    value="Chat",
                    elem_id="mode-radio"
                )
                with gr.Accordion("Voice Settings", open=False):
                    generate_voice_checkbox = gr.Checkbox(label="Generate Voice", value=False)
                    voice_dropdown = gr.Dropdown(
                        choices=get_available_voices(),
                        label="Select Voice",
                        value=get_available_voices()[0] if get_available_voices() else None,
                    )
                audio_output = gr.Audio(label="Voice Output", visible=False, autoplay=True)

        with gr.Tabs() as tabs:
            with gr.TabItem("Code Generation", id="code-tab"):
                with gr.Row():
                    coding_model_dropdown = gr.Dropdown(
                        choices=get_available_models(),
                        label="Select Coding Model",
                        value=get_available_models()[0] if get_available_models() else None,
                    )
                code_output = gr.Code(label="Code Output", language="python", elem_classes="code-area")
                with gr.Row():
                    prev_button = gr.Button("◀ Previous", size="sm")
                    next_button = gr.Button("Next ▶", size="sm")
                    download_code_button = gr.Button("💾 Download", size="sm")
                    copy_button = gr.Button("📋 Copy", size="sm")
                with gr.Row():
                    execute_button = gr.Button("▶️ Execute", size="sm")
                    lint_button = gr.Button("🔍 Lint", size="sm")
                code_status = gr.Textbox(label="Status", interactive=False)

            with gr.TabItem("Settings", id="settings-tab"):
                with gr.Row():
                    with gr.Column():
                        max_length = gr.Slider(50, 4200, value=250, step=10, label="Max Length")
                        temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    with gr.Column():
                        top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k")
                        top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                num_sequences = gr.Slider(1, 5, value=1, step=1, label="Number of Sequences")
                
                with gr.Row():
                    image_input = gr.Image(type="filepath", label="Upload Image for Analysis")
                    document_input = gr.File(label="Upload Document for Context (RAG)")

            with gr.TabItem("Sessions", id="sessions-tab"):
                with gr.Row():
                    new_session_button = gr.Button("New Session 🆕", size="sm")
                    session_dropdown = gr.Dropdown(choices=[], label="Select Session")
                    load_session_button = gr.Button("Load Session 📂", size="sm")
                    delete_session_button = gr.Button("Delete Session 🗑️", size="sm")

            with gr.TabItem("Model Management", id="model-tab"):
                with gr.Row():
                    model_name_input = gr.Textbox(label="Model Name")
                    download_model_button = gr.Button("Download Model", size="sm")
                with gr.Row():
                    delete_model_dropdown = gr.Dropdown(
                        choices=get_available_models(), label="Select Model to Delete"
                    )
                    delete_model_button = gr.Button("Delete Selected Model", size="sm")
                with gr.Row():
                    load_model_dropdown = gr.Dropdown(
                        choices=get_available_models(), label="Select Model to Load Modelfile"
                    )
                    load_modelfile_button = gr.Button("Load Modelfile", size="sm")
                modelfile_input = gr.TextArea(lines=10, label="Modelfile Content")
                save_modelfile_button = gr.Button("Save Modelfile", size="sm")
                model_management_output = gr.Textbox(label="Output", lines=5)

            with gr.TabItem("Library Explorer", id="library-tab"):
                with gr.Row():
                    module_list = gr.Dropdown(choices=explore_modules(), label="Available Modules")
                    import_button = gr.Button("Import Selected Module", size="sm")
                import_output = gr.Textbox(label="Import Output", lines=2)

        # Event handlers
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
            ],
            outputs=[chat_display, code_output, chat_display, audio_output, code_output, input_text],
        )

        prev_button.click(lambda: cycle_markdown("prev"), outputs=[code_output])
        next_button.click(lambda: cycle_markdown("next"), outputs=[code_output])

        mic_button.click(record_audio, outputs=[input_text])

        new_session_button.click(new_session, outputs=[session_dropdown, chat_display, code_output])
        load_session_button.click(
            load_session, inputs=[session_dropdown], outputs=[chat_display, code_output]
        )
        delete_session_button.click(
            delete_session, inputs=[session_dropdown], outputs=[session_dropdown, chat_display, code_output]
        )

        generate_voice_checkbox.change(
            lambda x: gr.update(visible=x), inputs=[generate_voice_checkbox], outputs=[audio_output]
        )

        download_code_button.click(download_code, inputs=[code_output], outputs=[gr.File()])
        copy_button.click(lambda x: x, inputs=[code_output], outputs=[gr.Textbox(visible=False)])

        download_model_button.click(
            download_model, inputs=[model_name_input], outputs=[model_management_output]
        )
        delete_model_button.click(
            delete_model, inputs=[delete_model_dropdown], outputs=[model_management_output]
        )
        load_modelfile_button.click(
            load_modelfile, inputs=[load_model_dropdown], outputs=[modelfile_input]
        )
        save_modelfile_button.click(
            save_modelfile, inputs=[model_name_input, modelfile_input], outputs=[model_management_output]
        )

        # Update model lists after download or delete operations
        download_model_button.click(
            update_model_list,
            outputs=[
                main_model_dropdown,
                coding_model_dropdown,
                delete_model_dropdown,
                load_model_dropdown,
            ],
        )
        delete_model_button.click(
            update_model_list,
            outputs=[
                main_model_dropdown,
                coding_model_dropdown,
                delete_model_dropdown,
                load_model_dropdown,
            ],
        )

        # Code execution and linting
        execute_button.click(execute_code, inputs=[code_output], outputs=[code_status])
        lint_button.click(lint_code, inputs=[code_output], outputs=[code_status])

        # Module import
        import_button.click(import_module, inputs=[module_list], outputs=[import_output])

        # New handler for clearing chat
        clear_button.click(
            lambda: ("", ""),
            outputs=[chat_display, input_text]
        )

    return iface

if __name__ == "__main__":
    try:
        iface = create_interface()
        iface.launch(share=False, server_name="127.0.0.1")
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")
        print(f"An error occurred while launching the interface: {e}")
        traceback.print_exc()
