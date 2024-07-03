import gradio as gr
import requests
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ollama API URL (local)
OLLAMA_API_URL = "http://localhost:11434/api"

# Chat history and sessions
chat_history = []
markdown_history = []
current_markdown_index = 0
sessions = {}
current_session = None

def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        response.raise_for_status()
        models = json.loads(response.text)
        return [model['name'] for model in models['models']]
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return ["No models found"]

def generate_text(model, prompt, max_length, temperature, top_k, top_p, num_sequences, image=None, context=None):
    data = {
        "model": model,
        "prompt": prompt,
        "options": {
            "num_predict": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    }

    if image:
        try:
            with open(image, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            data["images"] = [base64_image]
        except Exception as e:
            logger.error(f"Error processing image: {e}")

    if context:
        data["prompt"] = f"Context: {context}\n\nPrompt: {prompt}"

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_sequences) as executor:
            futures = [executor.submit(send_ollama_request, data) for _ in range(num_sequences)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return "\n\n--- New Sequence ---\n\n".join(responses)
    except Exception as e:
        logger.error(f"Error in generate_text: {e}")
        return f"An error occurred: {str(e)}"

def send_ollama_request(data):
    try:
        response = requests.post(f"{OLLAMA_API_URL}/generate", json=data)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        full_response = "".join(json.loads(line)['response'] for line in lines if 'response' in json.loads(line))
        return full_response
    except Exception as e:
        logger.error(f"Error in Ollama request: {e}")
        return f"Error: {str(e)}"

def extract_text_from_document(file):
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

def get_available_voices():
    engine = pyttsx3.init()
    return [voice.name for voice in engine.getProperty('voices')]

def text_to_speech(text, voice_name):
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

def generate_with_context(main_model, coding_model, prompt, max_length, temperature, top_k, top_p, num_sequences, image, document, mode, voice_name, generate_voice):
    global chat_history, markdown_history, current_markdown_index
    context = extract_text_from_document(document) if document else None
    if chat_history:
        context = (context or "") + "\n\nChat History:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    main_response = generate_text(main_model, prompt, max_length, temperature, top_k, top_p, num_sequences, image, context)
    
    chat_history.append({"role": "user", "content": prompt})
    
    audio_output = None
    if generate_voice:
        audio_output = text_to_speech(main_response, voice_name)
    
    if mode == "Coding":
        # Generate explanation and proposed changes
        explanation_prompt = f"Context: {context}\n\nUser Request: {prompt}\n\nProvide an explanation of the proposed changes and how to use them. Do not include any code in this response."
        explanation = generate_text(main_model, explanation_prompt, max_length, temperature, top_k, top_p, 1)
        
        # Generate example code
        example_prompt = f"Context: {context}\n\nUser Request: {prompt}\n\nGenerate example code based on the request. Only include the code, no explanations."
        example_code = generate_text(main_model, example_prompt, max_length, temperature, top_k, top_p, 1)
        
        # Send to coding model
        coding_prompt = f"Context: {context}\n\nUser Request: {prompt}\n\nMain Model Explanation: {explanation}\n\nMain Model Example Code:\n{example_code}\n\nGenerate the final code based on the above context, explanation, and example. Only output the code, no explanations."
        code_response = generate_text(coding_model, coding_prompt, max_length, temperature, top_k, top_p, 1)
        
        chat_history.append({"role": "assistant", "content": explanation})
        markdown_history.append(code_response)
        current_markdown_index = len(markdown_history) - 1
        return chat_history_to_string(), code_response, chat_history_to_string(), audio_output, gr.update(value=code_response)
    else:
        chat_history.append({"role": "assistant", "content": main_response})
        return chat_history_to_string(), "", chat_history_to_string(), audio_output, gr.update(value="")

def continue_code_generation(coding_model, current_code, max_length, temperature, top_k, top_p):
    continuation_prompt = f"Continue the following code:\n\n{current_code}\n\nContinue from here:"
    continuation = generate_text(coding_model, continuation_prompt, max_length, temperature, top_k, top_p, 1)
    full_code = current_code + "\n" + continuation
    markdown_history.append(full_code)
    global current_markdown_index
    current_markdown_index = len(markdown_history) - 1
    return gr.update(value=full_code)

def chat_history_to_string():
    chat_html = '<div style="display: flex; flex-direction: column; gap: 15px; font-size: 16px; width: 100%;">'
    for i, msg in enumerate(chat_history):
        if msg['role'] == 'user':
            chat_html += f'''
            <div style="align-self: flex-end; max-width: 80%; background-color: #1982FC; color: white; padding: 12px 18px; border-radius: 20px 20px 0 20px; position: relative; word-wrap: break-word;">
                <div style="font-size: 1.1em;">{msg["content"]}</div>
                <div style="font-size: 0.8em; opacity: 0.7; text-align: right; margin-top: 5px;">You</div>
            </div>
            '''
        else:
            chat_html += f'''
            <div style="align-self: flex-start; max-width: 80%; background-color: #34C759; color: white; padding: 12px 18px; border-radius: 20px 20px 20px 0; position: relative; word-wrap: break-word;">
                <div style="font-size: 1.1em;">{msg["content"]}</div>
                <div style="font-size: 0.8em; opacity: 0.7; margin-top: 5px;">Assistant</div>
            </div>
            '''
    chat_html += '</div>'
    return chat_html

def cycle_markdown(direction):
    global current_markdown_index
    if direction == "next" and current_markdown_index < len(markdown_history) - 1:
        current_markdown_index += 1
    elif direction == "prev" and current_markdown_index > 0:
        current_markdown_index -= 1
    return markdown_history[current_markdown_index]

def markdown_history_to_string():
    return "\n\n---\n\n".join(markdown_history)

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def new_session():
    global current_session, chat_history, markdown_history
    if current_session:
        sessions[current_session] = (chat_history, markdown_history)
    current_session = f"Session {len(sessions) + 1}"
    chat_history = []
    markdown_history = []
    return gr.update(choices=list(sessions.keys()) + [current_session], value=current_session), "", ""

def load_session(session_name):
    global current_session, chat_history, markdown_history
    if session_name in sessions:
        current_session = session_name
        chat_history, markdown_history = sessions[session_name]
    return chat_history_to_string(), markdown_history_to_string()

def delete_session(session_name):
    global current_session, chat_history, markdown_history
    if session_name in sessions:
        del sessions[session_name]
        if current_session == session_name:
            current_session = None
            chat_history = []
            markdown_history = []
    return gr.update(choices=list(sessions.keys())), "", ""

def download_code(code):
    with open("code_output.txt", "w") as f:
        f.write(code)
    return "code_output.txt"

def download_model(model_name):
    try:
        result = subprocess.run(['ollama', 'run', model_name], capture_output=True, text=True)
        return f"Model {model_name} downloaded successfully."
    except subprocess.CalledProcessError as e:
        return f"Error downloading model: {e.output}"

def delete_model(model_name):
    try:
        result = subprocess.run(['ollama', 'rm', model_name], capture_output=True, text=True)
        return f"Model {model_name} deleted successfully."
    except subprocess.CalledProcessError as e:
        return f"Error deleting model: {e.output}"

def load_modelfile(model_name):
    try:
        result = subprocess.run(['ollama', 'show', model_name], capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error loading modelfile: {e.output}"

def save_modelfile(model_name, modelfile_content):
    try:
        with open(f"{model_name}.modelfile", "w") as f:
            f.write(modelfile_content)
        result = subprocess.run(['ollama', 'create', model_name, '-f', f"{model_name}.modelfile"], capture_output=True, text=True)
        os.remove(f"{model_name}.modelfile")
        return f"Modelfile for {model_name} saved and applied successfully."
    except Exception as e:
        return f"Error saving modelfile: {str(e)}"

# Gradio interface setup
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
                mode_radio = gr.Radio(["Chat", "Coding"], label="Mode", value="Chat")
                generate_voice_checkbox = gr.Checkbox(label="Generate Voice", value=False)
            voice_dropdown = gr.Dropdown(choices=get_available_voices(), label="Select Voice", value=get_available_voices()[0] if get_available_voices() else None)
            audio_output = gr.Audio(label="Voice Output", visible=False)
        
        with gr.Column(scale=2):
            coding_model_dropdown = gr.Dropdown(choices=get_available_models(), label="Select Coding Model", value=get_available_models()[0] if get_available_models() else None)
            code_output = gr.Code(label="Code Output", language="python")
            with gr.Row():
                prev_button = gr.Button("◀ Previous")
                next_button = gr.Button("Next ▶")
                download_button = gr.Button("Download Code")
                copy_button = gr.Button("Copy to Clipboard")
            continue_button = gr.Button("Continue Generation")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Image")
            document_input = gr.File(label="Upload Document for Context (RAG)")
        with gr.Column(scale=2):
            max_length = gr.Slider(50, 42000, value=250, step=10, label="Max Length")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
            top_k = gr.Slider(0, 100, value=50, step=1, label="Top-k")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p (nucleus sampling)")
            num_sequences = gr.Slider(1, 5, value=1, step=1, label="Number of Sequences")

    with gr.Row():
        new_session_button = gr.Button("New Session")
        session_dropdown = gr.Dropdown(choices=[], label="Load Session")
        load_session_button = gr.Button("Load Selected Session")
        delete_session_button = gr.Button("Delete Selected Session")

    generate_button.click(
        generate_with_context,
        inputs=[main_model_dropdown, coding_model_dropdown, input_text, max_length, temperature, top_k, top_p, num_sequences, image_input, document_input, mode_radio, voice_dropdown, generate_voice_checkbox],
        outputs=[chat_display, code_output, chat_display, audio_output, code_output]
    )

    continue_button.click(
        continue_code_generation,
        inputs=[coding_model_dropdown, code_output, max_length, temperature, top_k, top_p],
        outputs=[code_output]
    )

    prev_button.click(lambda: cycle_markdown("prev"), outputs=[code_output])
    next_button.click(lambda: cycle_markdown("next"), outputs=[code_output])

    mic_button.click(record_audio, outputs=input_text)

    new_session_button.click(new_session, outputs=[session_dropdown, chat_display, code_output])
    load_session_button.click(load_session, inputs=[session_dropdown], outputs=[chat_display, code_output])
    delete_session_button.click(delete_session, inputs=[session_dropdown], outputs=[session_dropdown, chat_display, code_output])

    generate_voice_checkbox.change(lambda x: gr.update(visible=x), inputs=[generate_voice_checkbox], outputs=[audio_output])

    download_button.click(download_code, inputs=[code_output], outputs=[gr.File()])
    copy_button.click(lambda x: gr.update(value=x), inputs=[code_output], outputs=[gr.Textbox(visible=False)])

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

        delete_button.click(delete_model, inputs=[delete_model_dropdown], outputs=[model_management_output])
        load_modelfile_button.click(load_modelfile, inputs=[load_model_dropdown], outputs=[modelfile_input])
        save_modelfile_button.click(save_modelfile, inputs=[model_name_input, modelfile_input], outputs=[model_management_output])

    gr.Markdown("""
    ## Parameter Explanations:
    - **Max Length**: The maximum number of tokens in the generated text.
    - **Temperature**: Controls randomness. Lower values make the output more focused and deterministic.
    - **Top-k**: Limits the next token selection to the k most probable tokens.
    - **Top-p (nucleus sampling)**: Dynamically selects the smallest set of tokens whose cumulative probability exceeds p.
    """)

if __name__ == "__main__":
    try:
        iface.launch(share=False, server_name="127.0.0.1")
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")
        print(f"An error occurred while launching the interface: {e}")
        import traceback
        traceback.print_exc()

# Add custom CSS and JavaScript for chat bubbles and full-width layout
iface.load(css="""
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
        background-color: ##34C759;
        align-self: flex-end;
        border-bottom-right-radius: 20px;
        border-bottom-left-radius: 5px;
    }
    .assistant {
        background-color: #34C759;  /* Changed to green */
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
""", js="""
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
""")
