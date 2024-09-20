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
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
import threading
import time
import pkgutil
import importlib

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
    context = extract_text_from_document(document) if document else None
    if chat_history:
        previous_conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        context = f"{context}\n\nChat History:\n{previous_conversation}" if context else previous_conversation

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
    chat_html = '<div style="display: flex; flex-direction: column; gap: 15px; font-size: 16px; width: 100%;">'
    for msg in chat_history:
        style = (
            "align-self: flex-end; background-color: #1982FC;"
            if msg['role'] == 'user'
            else "align-self: flex-start; background-color: #34C759;"
        )
        chat_html += f'''
        <div style="{style} max-width: 80%; color: white; padding: 12px 18px; border-radius: 20px; position: relative; word-wrap: break-word;">
            <div style="font-size: 1.1em;">{msg["content"]}</div>
            <div style="font-size: 0.8em; opacity: 0.7; {'text-align: right; ' if msg['role'] == 'user' else ''}margin-top: 5px;">{msg['role'].capitalize()}</div>
        </div>
        '''
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
    # Placeholder for document processing
    return "Document QA functionality is under development."

# Gradio interface setup
def create_interface():
    with gr.Blocks(title="Enhanced Ollama Text Generation") as iface:
        gr.Markdown("# Enhanced Ollama Text Generation")

        with gr.Row():
            with gr.Column(scale=3):
                main_model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Main Model",
                    value=get_available_models()[0] if get_available_models() else None,
                )
                chat_display = gr.HTML(label="Chat History")
                with gr.Row():
                    input_text = gr.Textbox(
                        lines=2, label="Input Prompt", placeholder="Type your message here..."
                    )
                    mic_button = gr.Button("🎤")
                    generate_button = gr.Button("Generate")
                with gr.Row():
                    mode_radio = gr.Radio(
                        ["Chat", "Coding", "Document QA"], label="Mode", value="Chat"
                    )
                    generate_voice_checkbox = gr.Checkbox(label="Generate Voice", value=False)
                voice_dropdown = gr.Dropdown(
                    choices=get_available_voices(),
                    label="Select Voice",
                    value=get_available_voices()[0] if get_available_voices() else None,
                )
                audio_output = gr.Audio(label="Voice Output", visible=False, autoplay=True)

            with gr.Column(scale=2):
                coding_model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Coding Model",
                    value=get_available_models()[0] if get_available_models() else None,
                )
                code_output = gr.Code(label="Code Output", language="python")
                with gr.Row():
                    prev_button = gr.Button("◀ Previous")
                    next_button = gr.Button("Next ▶")
                    download_code_button = gr.Button("Download Code")
                    copy_button = gr.Button("Copy to Clipboard")
                with gr.Row():
                    execute_button = gr.Button("Execute Code")
                    lint_button = gr.Button("Lint Code")
                code_status = gr.Textbox(label="Code Generation Status", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image")
                document_input = gr.File(label="Upload Document for Context (RAG)")
            with gr.Column(scale=2):
                max_length = gr.Slider(50, 4200, value=250, step=10, label="Max Length")
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                num_sequences = gr.Slider(1, 5, value=1, step=1, label="Number of Sequences")

        with gr.Row():
            new_session_button = gr.Button("New Session")
            session_dropdown = gr.Dropdown(choices=[], label="Load Session")
            load_session_button = gr.Button("Load Selected Session")
            delete_session_button = gr.Button("Delete Selected Session")

        with gr.Tab("Model Management"):
            with gr.Row():
                model_name_input = gr.Textbox(label="Model Name")
                download_model_button = gr.Button("Download Model")
                delete_model_dropdown = gr.Dropdown(
                    choices=get_available_models(), label="Select Model to Delete"
                )
                delete_model_button = gr.Button("Delete Selected Model")
            with gr.Row():
                load_model_dropdown = gr.Dropdown(
                    choices=get_available_models(), label="Select Model to Load Modelfile"
                )
                load_modelfile_button = gr.Button("Load Modelfile")
            with gr.Row():
                modelfile_input = gr.TextArea(lines=10, label="Modelfile Content")
                save_modelfile_button = gr.Button("Save Modelfile")
            model_management_output = gr.Textbox(label="Output", lines=5)

        with gr.Tab("Library Explorer"):
            with gr.Row():
                module_list = gr.Dropdown(choices=explore_modules(), label="Available Modules")
                import_button = gr.Button("Import Selected Module")
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

        gr.Markdown(
            """
        ## Parameter Explanations:
        - **Max Length**: The maximum number of tokens in the generated text.
        - **Temperature**: Controls randomness. Lower values make the output more focused and deterministic.
        - **Top-k**: Limits the next token selection to the k most probable tokens.
        - **Top-p**: Dynamically selects the smallest set of tokens whose cumulative probability exceeds p.
        - **Number of Sequences**: The number of alternative completions to generate.
        """
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
