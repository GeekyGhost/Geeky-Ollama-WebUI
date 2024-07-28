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
from tqdm import tqdm
import re
import ast
import pylint.lint
from pylint.lint import Run
from pylint.reporters.text import TextReporter
import io
import sys
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from RestrictedPython.PrintCollector import PrintCollector
import pkgutil
import importlib
import jedi
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OLLAMA_API_URL = "http://localhost:11434/v1"

# Global variables
chat_history: List[Dict[str, str]] = []
markdown_history: List[str] = []
current_markdown_index: int = 0
sessions: Dict[str, Tuple[List[Dict[str, str]], List[str]]] = {}
current_session: Optional[str] = None
code_versions: List[str] = []

# Create an Ollama client
client = ollama.Client(host=OLLAMA_API_URL)

def get_available_models() -> List[str]:
    try:
        models = client.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return ["No models found"]

def generate_text(model: str, prompt: str, max_length: int, temperature: float, top_k: int, top_p: float, num_sequences: int, image: Optional[str] = None, context: Optional[str] = None) -> str:
    messages = [{'role': 'user', 'content': prompt}]
    if context:
        messages.insert(0, {'role': 'system', 'content': context})

    options = {
        'num_predict': max_length,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
    }

    if image:
        try:
            with open(image, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            messages[0]['images'] = [base64_image]
        except IOError as e:
            logger.error(f"Error processing image: {e}")

    try:
        responses = []
        for _ in tqdm(range(num_sequences), desc="Generating sequences", unit="seq"):
            response = client.chat(model=model, messages=messages, options=options, stream=True)
            full_response = ""
            for chunk in tqdm(response, desc="Processing response", leave=False):
                full_response += chunk['message']['content']
            
            full_response = full_response.replace("```python", "").replace("```", "").strip()
            responses.append(full_response)
        
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

def generate_with_context(main_model: str, coding_model: str, prompt: str, max_length: int, temperature: float, top_k: int, top_p: float, num_sequences: int, image, document, mode: str, voice_name: str, generate_voice: bool):
    global chat_history, markdown_history, current_markdown_index
    context = extract_text_from_document(document) if document else None
    if chat_history:
        context = (context or "") + "\n\nChat History:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    if mode == "Coding":
        code_snippets = extract_code_snippets(context) if context else []
        context_with_snippets = (context or "") + "\n\nRelevant Code Snippets:\n" + "\n".join(code_snippets)
        
        explanation_prompt = f"Context: {context_with_snippets}\n\nUser Request: {prompt}\n\nProvide an explanation of the proposed changes and how to use them. Do not include any code in this response."
        explanation = generate_text(main_model, explanation_prompt, max_length, temperature, top_k, top_p, 1)
        
        example_prompt = f"Context: {context_with_snippets}\n\nUser Request: {prompt}\n\nGenerate example code based on the request. Only include the code, no explanations."
        example_code = generate_text(main_model, example_prompt, max_length, temperature, top_k, top_p, 1)
        
        coding_prompt = f"Generate Python code based on the following request. Only output code with proper comments. Do not include any explanations outside of code comments.\n\nContext: {context_with_snippets}\n\nUser Request: {prompt}\n\nMain Model Explanation: {explanation}\n\nMain Model Example Code:\n{example_code}"
        code_response = generate_text(coding_model, coding_prompt, max_length, temperature, top_k, top_p, 1)
        
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": explanation})
        markdown_history.append(code_response)
        current_markdown_index = len(markdown_history) - 1
        code_versions.append(code_response)
        return chat_history_to_string(), code_response, chat_history_to_string(), None, gr.update(value=code_response), gr.update(value="")
    else:
        main_response = generate_text(main_model, prompt, max_length, temperature, top_k, top_p, num_sequences, image, context)
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": main_response})
        
        audio_output = None
        if generate_voice:
            audio_output = text_to_speech(main_response, voice_name)
        
        return chat_history_to_string(), "", chat_history_to_string(), audio_output, gr.update(value=""), gr.update(value="")

def continue_code_generation(coding_model: str, current_code: str, user_request: str, max_length: int, temperature: float, top_k: int, top_p: float) -> Tuple[gr.update, str]:
    try:
        # Try to parse the current code
        try:
            tree = ast.parse(current_code)
        except IndentationError as ie:
            # If there's an indentation error, try to fix it
            lines = current_code.split('\n')
            fixed_lines = []
            for line in lines:
                if line.strip():  # If the line is not empty
                    fixed_lines.append(line.strip())  # Remove leading/trailing whitespace
                else:
                    fixed_lines.append('')  # Keep empty lines
            fixed_code = '\n'.join(fixed_lines)
            tree = ast.parse(fixed_code)
            current_code = fixed_code
        
        # Find the last complete code block (class, function, or main code)
        last_node = tree.body[-1] if tree.body else None
        last_block_start = 0
        if last_node:
            last_block_start = last_node.lineno - 1
        
        # Extract the context (last few lines of the last complete block)
        lines = current_code.split('\n')
        context_lines = lines[last_block_start:]
        context = '\n'.join(context_lines)
        
        # Implement a sliding context window
        max_context_lines = 50  # Adjust this value as needed
        if len(context_lines) > max_context_lines:
            context = '\n'.join(context_lines[-max_context_lines:])
        
        # Determine the indentation of the last line
        last_non_empty_line = next((line for line in reversed(lines) if line.strip()), '')
        indentation = len(last_non_empty_line) - len(last_non_empty_line.lstrip())
        
        continuation_prompt = f"""Continue the following Python code. Analyze the existing code structure and continue from the last complete block or statement. Maintain the current class structure, function implementations, and coding style. Only output code with proper comments. Do not include any explanations outside of code comments. Do not repeat any existing code.

User Request: {user_request}

Current Code Context (last {max_context_lines} lines or less):
{context}

Continue from here, maintaining the appropriate indentation and structure:
{' ' * indentation}"""
        
        continuation = generate_text(coding_model, continuation_prompt, max_length, temperature, top_k, top_p, 1)
        
        # Remove any leading whitespace or newlines
        continuation = continuation.lstrip()
        
        # Ensure the continuation starts with the correct indentation
        if not continuation.startswith(' ' * indentation):
            continuation = ' ' * indentation + continuation
        
        # Combine the current code with the continuation
        full_code = '\n'.join(lines[:last_block_start]) + '\n' + context + '\n' + continuation
        
        # Verify the generated code
        try:
            ast.parse(full_code)
        except SyntaxError as se:
            return gr.update(value=full_code), f"Generated code has a syntax error: {str(se)}"
        except IndentationError as ie:
            return gr.update(value=full_code), f"Generated code has an indentation error: {str(ie)}"
        
        markdown_history.append(full_code)
        global current_markdown_index
        current_markdown_index = len(markdown_history) - 1
        code_versions.append(full_code)
        
        status = "Code continuation generated successfully."
        return gr.update(value=full_code), status
    except Exception as e:
        error_message = f"Error occurred while generating continuation: {str(e)}\n{traceback.format_exc()}"
        return gr.update(value=current_code), error_message

def refactor_code(coding_model: str, current_code: str, user_request: str, max_length: int, temperature: float, top_k: int, top_p: float) -> Tuple[gr.update, str]:
    # Analyze the current code
    tree = ast.parse(current_code)
    
    # Identify potential refactoring opportunities
    refactoring_opportunities = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if len(node.body) > 20:  # Long function, might need extraction
                refactoring_opportunities.append(f"Consider extracting parts of the function '{node.name}' into smaller functions.")
        elif isinstance(node, ast.For) or isinstance(node, ast.While):
            if len(node.body) > 15:  # Long loop, might need optimization
                refactoring_opportunities.append("Consider optimizing the loop starting at line {node.lineno}.")
    
    refactor_prompt = f"""Refactor the following Python code. Review the user request and the code generated so far, then provide a refactored version. Consider the following refactoring opportunities:
{chr(10).join(refactoring_opportunities)}

Only output code with proper comments. Do not include any explanations outside of code comments. Do not use markdown code block markers.

User Request: {user_request}

Current Code:
{current_code}

Refactored Code:
"""
    
    try:
        refactored_code = generate_text(coding_model, refactor_prompt, max_length, temperature, top_k, top_p, 1)
        
        # Remove any leading whitespace or newlines
        refactored_code = refactored_code.lstrip()
        
        markdown_history.append(refactored_code)
        global current_markdown_index
        current_markdown_index = len(markdown_history) - 1
        code_versions.append(refactored_code)
        
        status = "Code refactored successfully."
        return gr.update(value=refactored_code), status
    except Exception as e:
        logger.error(f"Error in refactor_code: {e}")
        error_message = f"Error occurred while refactoring code: {str(e)}"
        return gr.update(value=current_code + f"\n\n# {error_message}"), error_message

def chat_history_to_string() -> str:
    chat_html = '<div style="display: flex; flex-direction: column; gap: 15px; font-size: 16px; width: 100%;">'
    for msg in chat_history:
        style = "align-self: flex-end; background-color: #1982FC;" if msg['role'] == 'user' else "align-self: flex-start; background-color: #34C759;"
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

def markdown_history_to_string() -> str:
    return "\n\n---\n\n".join(markdown_history)

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
    for i in progress.tqdm(range(100)):
        if i >= 99:  # Stop after 10 seconds
            recording = False
        time.sleep(0.1)
    
    recording = False
    thread.join()
    
    full_audio = sr.AudioData(b''.join(chunk.frame_data for chunk in audio_data),
                              audio_data[0].sample_rate,
                              audio_data[0].sample_width)
    
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
    return gr.update(choices=list(sessions.keys()) + [current_session], value=current_session), "", ""

def load_session(session_name: str) -> Tuple[str, str]:
    global current_session, chat_history, markdown_history
    if session_name in sessions:
        current_session = session_name
        chat_history, markdown_history = sessions[session_name]
    return chat_history_to_string(), markdown_history_to_string()

def delete_session(session_name: str) -> Tuple[gr.update, str, str]:
    global current_session, chat_history, markdown_history
    if session_name in sessions:
        del sessions[session_name]
        if current_session == session_name:
            current_session = None
            chat_history = []
            markdown_history = []
    return gr.update(choices=list(sessions.keys())), "", ""

def download_code(code: str) -> str:
    with open("code_output.txt", "w") as f:
        f.write(code)
    return "code_output.txt"

def download_model(model_name: str) -> str:
    try:
        client.pull(model_name)
        return f"Model {model_name} downloaded successfully."
    except Exception as e:
        return f"Error downloading model: {str(e)}"

def delete_model(model_name: str) -> str:
    try:
        client.delete(model_name)
        return f"Model {model_name} deleted successfully."
    except Exception as e:
        return f"Error deleting model: {str(e)}"

def load_modelfile(model_name: str) -> str:
    try:
        return client.show(model_name)['modelfile']
    except Exception as e:
        return f"Error loading modelfile: {str(e)}"

def save_modelfile(model_name: str, modelfile_content: str) -> str:
    try:
        with open(f"{model_name}.modelfile", "w") as f:
            f.write(modelfile_content)
        result = subprocess.run(['ollama', 'create', model_name, '-f', f"{model_name}.modelfile"], capture_output=True, text=True, check=True)
        os.remove(f"{model_name}.modelfile")
        return f"Modelfile for {model_name} saved and applied successfully."
    except Exception as e:
        return f"Error saving modelfile: {str(e)}"

def update_model_list():
    return gr.update(choices=get_available_models())

def markdown_to_code(markdown_content: str) -> str:
    code_blocks = re.findall(r'```python\n(.*?)```', markdown_content, re.DOTALL)
    return '\n\n'.join(code_blocks)

def code_to_markdown(code: str) -> str:
    return f"```python\n{code}\n```"

def execute_code(code: str) -> str:
    try:
        # Create a restricted environment
        restricted_globals = safe_builtins.copy()
        restricted_globals['_print_'] = PrintCollector
        restricted_globals['__builtins__'] = restricted_globals

        # Compile and execute the code
        byte_code = compile_restricted(code, '<string>', 'exec')
        exec(byte_code, restricted_globals)

        # Get the printed output
        output = restricted_globals['_print']()
        return output
    except Exception as e:
        return f"Error executing code: {str(e)}"

def lint_code(code: str) -> str:
    try:
        tree = ast.parse(code)
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 5:
                    issues.append(f"Line {node.lineno}: Function '{node.name}' has more than 5 parameters.")
            elif isinstance(node, ast.Try):
                if not node.handlers:
                    issues.append(f"Line {node.lineno}: Try block without except handlers.")
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                if any(alias.asname for alias in node.names):
                    issues.append(f"Line {node.lineno}: Consider avoiding 'import as' for clarity.")

        if issues:
            return "\n".join(issues)
        else:
            return "No issues found."
    except SyntaxError as e:
        return f"Syntax error: {str(e)}"
    except Exception as e:
        return f"An error occurred during code analysis: {str(e)}"

def get_autocomplete_suggestions(code: str, line: int, column: int) -> List[str]:
    script = jedi.Script(code)
    completions = script.complete(line, column)
    return [c.name for c in completions]

def explore_modules() -> List[str]:
    return [name for _, name, _ in pkgutil.iter_modules()]

def import_module(module_name: str) -> str:
    try:
        importlib.import_module(module_name)
        return f"Module {module_name} imported successfully."
    except ImportError as e:
        return f"Error importing module {module_name}: {str(e)}"

def process_document(file_path: str, question: str) -> str:
    # Load the document
    loader = WebBaseLoader(file_path)
    data = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    all_splits = text_splitter.split_documents(data)

    # Create embeddings and store in vector database
    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

    # Create a question-answering chain
    qa_chain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

    # Ask the question
    result = qa_chain.invoke({"query": question})
    return result['result']

# Gradio interface setup
def create_interface():
    with gr.Blocks(title="Enhanced Ollama Text Generation", 
                   theme=gr.themes.Base(),
                   css="footer {display:none} .container {max-width: 100% !important; padding: 0 !important;}") as iface:
        gr.Markdown("# Enhanced Ollama Text Generation")
        
        with gr.Row():
            with gr.Column(scale=3):
                main_model_dropdown = gr.Dropdown(choices=get_available_models(), label="Select Main Model", value=get_available_models()[0] if get_available_models() else None)
                chat_display = gr.HTML(label="Chat History")
                with gr.Row():
                    input_text = gr.Textbox(lines=2, label="Input Prompt", placeholder="Type your message here...", scale=20)
                    mic_button = gr.Button("🎤", scale=1)
                    generate_button = gr.Button("Generate", scale=3)
                with gr.Row():
                    mode_radio = gr.Radio(["Chat", "Coding", "Document QA"], label="Mode", value="Chat")
                    generate_voice_checkbox = gr.Checkbox(label="Generate Voice", value=False)
                voice_dropdown = gr.Dropdown(choices=get_available_voices(), label="Select Voice", value=get_available_voices()[0] if get_available_voices() else None)
                audio_output = gr.Audio(label="Voice Output", visible=False, autoplay=True)
            
            with gr.Column(scale=2):
                coding_model_dropdown = gr.Dropdown(choices=get_available_models(), label="Select Coding Model", value=get_available_models()[0] if get_available_models() else None)
                code_output = gr.Code(label="Code Output", language="python")
                code_editor = gr.TextArea(label="Code Editor", lines=10)
                with gr.Row():
                    prev_button = gr.Button("◀ Previous")
                    next_button = gr.Button("Next ▶")
                    download_button = gr.Button("Download Code")
                    copy_button = gr.Button("Copy to Clipboard")
                with gr.Row():
                    continue_button = gr.Button("Continue Generation")
                    refactor_button = gr.Button("Refactor Code")
                    execute_button = gr.Button("Execute Code")
                    lint_button = gr.Button("Lint Code")
                with gr.Row():
                    code_status = gr.Textbox(label="Code Generation Status", interactive=False)
                    execution_output = gr.Textbox(label="Execution Output", interactive=False)
                    lint_output = gr.Textbox(label="Lint Output", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image")
                document_input = gr.File(label="Upload Document for Context (RAG)")
            with gr.Column(scale=2):
                max_length = gr.Slider(50, 4200, value=250, step=10, label="Max Length")
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p (nucleus sampling)")
                num_sequences = gr.Slider(1, 5, value=1, step=1, label="Number of Sequences")

        with gr.Row():
            new_session_button = gr.Button("New Session")
            session_dropdown = gr.Dropdown(choices=[], label="Load Session")
            load_session_button = gr.Button("Load Selected Session")
            delete_session_button = gr.Button("Delete Selected Session")

        with gr.Tab("Model Management"):
            with gr.Row():
                model_name_input = gr.Textbox(label="Model Name")
                download_button = gr.Button("Download Model")
                delete_model_dropdown = gr.Dropdown(choices=get_available_models(), label="Select Model to Delete")
                delete_button = gr.Button("Delete Selected Model")
            
            with gr.Row():
                load_model_dropdown = gr.Dropdown(choices=get_available_models(), label="Select Model to Load Modelfile")
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
            inputs=[main_model_dropdown, coding_model_dropdown, input_text, max_length, temperature, top_k, top_p, num_sequences, image_input, document_input, mode_radio, voice_dropdown, generate_voice_checkbox],
            outputs=[chat_display, code_output, chat_display, audio_output, code_output, input_text]
        )

        continue_button.click(
            continue_code_generation,
            inputs=[coding_model_dropdown, code_output, input_text, max_length, temperature, top_k, top_p],
            outputs=[code_output, code_status]
        )

        refactor_button.click(
            refactor_code,
            inputs=[coding_model_dropdown, code_output, input_text, max_length, temperature, top_k, top_p],
            outputs=[code_output, code_status]
        )

        prev_button.click(partial(cycle_markdown, "prev"), outputs=[code_output])
        next_button.click(partial(cycle_markdown, "next"), outputs=[code_output])

        mic_button.click(
            record_audio,
            outputs=[input_text],
            show_progress=True
        )

        new_session_button.click(new_session, outputs=[session_dropdown, chat_display, code_output])
        load_session_button.click(load_session, inputs=[session_dropdown], outputs=[chat_display, code_output])
        delete_session_button.click(delete_session, inputs=[session_dropdown], outputs=[session_dropdown, chat_display, code_output])

        generate_voice_checkbox.change(lambda x: gr.update(visible=x), inputs=[generate_voice_checkbox], outputs=[audio_output])

        download_button.click(download_code, inputs=[code_output], outputs=[gr.File()])
        copy_button.click(lambda x: gr.update(value=x), inputs=[code_output], outputs=[gr.Textbox(visible=False)])

        download_button.click(download_model, inputs=[model_name_input], outputs=[model_management_output])
        delete_button.click(delete_model, inputs=[delete_model_dropdown], outputs=[model_management_output])
        load_modelfile_button.click(load_modelfile, inputs=[load_model_dropdown], outputs=[modelfile_input])
        save_modelfile_button.click(save_modelfile, inputs=[model_name_input, modelfile_input], outputs=[model_management_output])

        # Update model lists after download or delete operations
        download_button.click(update_model_list, outputs=[main_model_dropdown, coding_model_dropdown, delete_model_dropdown, load_model_dropdown])
        delete_button.click(update_model_list, outputs=[main_model_dropdown, coding_model_dropdown, delete_model_dropdown, load_model_dropdown])

        # Code editing
        code_output.change(lambda x: x, inputs=[code_output], outputs=[code_editor])
        code_editor.change(lambda x: x, inputs=[code_editor], outputs=[code_output])

        # Code execution and linting
        execute_button.click(execute_code, inputs=[code_output], outputs=[execution_output])
        lint_button.click(lint_code, inputs=[code_output], outputs=[lint_output])

        # Module import
        import_button.click(import_module, inputs=[module_list], outputs=[import_output])

        # Document QA
        def document_qa(document, question):
            if document and question:
                return process_document(document.name, question)
            return "Please upload a document and ask a question."

        mode_radio.change(
            lambda mode: gr.update(visible=mode == "Document QA"),
            inputs=[mode_radio],
            outputs=[document_input]
        )

        generate_button.click(
            document_qa,
            inputs=[document_input, input_text],
            outputs=[chat_display],
            show_progress=True
        )

        gr.Markdown("""
        ## Parameter Explanations:
        - **Max Length**: The maximum number of tokens in the generated text.
        - **Temperature**: Controls randomness. Lower values make the output more focused and deterministic.
        - **Top-k**: Limits the next token selection to the k most probable tokens.
        - **Top-p (nucleus sampling)**: Dynamically selects the smallest set of tokens whose cumulative probability exceeds p.
        - **Number of Sequences**: The number of alternative completions to generate.
        """)

    return iface

# Main execution
if __name__ == "__main__":
    try:
        iface = create_interface()
        iface.launch(share=False, server_name="127.0.0.1")

        # Apply custom CSS and JavaScript
        iface.load(css=custom_css, js=custom_js)
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")
        print(f"An error occurred while launching the interface: {e}")
        import traceback
        traceback.print_exc()

# CSS styles and JavaScript
custom_css = """
    .chat-container {
        display: flex;
        flex-direction: column;
        padding: 10px;
    }
    .chat-bubble {
        max-width: 70%;
        margin: 10px 0;
        padding: 10px;
        border-radius: 20px;
        position: relative;
        color: white;
        word-wrap: break-word;
    }
    .user {
        background-color: #1982FC;
        align-self: flex-end;
        border-bottom-right-radius: 20px;
        border-bottom-left-radius: 5px;
    }
    .assistant {
        background-color: #34C759;
        align-self: flex-start;
        border-bottom-left-radius: 20px;
        border-bottom-right-radius: 5px;
    }
    .chat-content {
        margin-bottom: 5px;
    }
    .chat-time {
        font-size: 0.75em;
        opacity: 0.7;
        margin-top: 5px;
    }
    .user .chat-time {
        text-align: right;
    }
    #chat-display {
        height: 500px;
        overflow-y: auto;
        background-color: #F0F0F0;
        border-radius: 10px;
        padding: 10px;
    }
    #code-output {
        height: 500px;
        overflow-y: auto;
    }
    #continue-button {
        margin-top: 10px;
    }
    .split-view {
        display: flex;
        height: 500px;
    }
    .split-view > div {
        flex: 1;
        overflow-y: auto;
        padding: 10px;
    }
    .collapsible {
        background-color: #f1f1f1;
        cursor: pointer;
        padding: 18px;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        font-size: 15px;
    }
    .active, .collapsible:hover {
        background-color: #e1e1e1;
    }
    .collapsible:after {
        content: '\\002B';
        font-weight: bold;
        float: right;
        margin-left: 5px;
    }
    .active:after {
        content: "\\2212";
    }
    .content {
        padding: 0 18px;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.2s ease-out;
        background-color: #f9f9f9;
    }
"""

# JavaScript for chat bubble toggling and code block collapsing
custom_js = """
    function toggleChatBubble(id) {
        const chatBubble = document.getElementById(`chat-${id}`);
        const content = chatBubble.querySelector('.chat-content');
        const toggleIcon = chatBubble.querySelector('.toggle-icon');
        
        if (content.style.display === 'none') {
            content.style.display = 'block';
            toggleIcon.textContent = '[-]';
        } else {
            content.style.display = 'none';
            toggleIcon.textContent = '[+]';
        }
    }

    var coll = document.getElementsByClassName("collapsible");
    var i;

    for (i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.maxHeight){
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
            }
        });
    }
"""

# Apply custom CSS and JavaScript
iface.load(css=custom_css, js=custom_js)
