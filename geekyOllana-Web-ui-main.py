import gradio as gr
import requests
import json
import base64
import os
from PyPDF2 import PdfReader
from docx import Document
import pyttsx3
import concurrent.futures
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ollama API URL (local)
OLLAMA_API_URL = "http://localhost:11434/api"

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

def text_to_speech(text, voice_name, auto_play):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        selected_voice = next((voice for voice in voices if voice.name == voice_name), voices[0])
        engine.setProperty('voice', selected_voice.id)

        output_file = "output.mp3"
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        
        return output_file if auto_play else None
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return None

def generate_with_context(model, prompt, max_length, temperature, top_k, top_p, num_sequences, image, document):
    context = extract_text_from_document(document) if document else None
    return generate_text(model, prompt, max_length, temperature, top_k, top_p, num_sequences, image, context)

def generate_and_speak(model, prompt, max_length, temperature, top_k, top_p, num_sequences, image, document, voice_name, auto_play):
    generated_text = generate_with_context(model, prompt, max_length, temperature, top_k, top_p, num_sequences, image, document)
    audio_output = text_to_speech(generated_text, voice_name, auto_play) if auto_play else None
    return generated_text, audio_output

# Gradio interface
with gr.Blocks(title="Local Ollama Text Generation with RAG and Vision") as iface:
    gr.Markdown("# Local Ollama Text Generation with RAG and Vision")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(choices=get_available_models(), label="Select Model", value=get_available_models()[0] if get_available_models() else None)
            image_input = gr.Image(type="filepath", label="Upload Image", height=300, width=300)
            document_input = gr.File(label="Upload Document for Context (RAG)")
        
        with gr.Column(scale=2):
            input_text = gr.Textbox(lines=5, label="Input Prompt")
            output_text = gr.Textbox(lines=15, label="Generated Text")
            audio_output = gr.Audio(label="Text-to-Speech Output", autoplay=True)
        
        with gr.Column(scale=1):
            max_length = gr.Slider(50, 1000, value=250, step=10, label="Max Length")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
            top_k = gr.Slider(0, 100, value=50, step=1, label="Top-k")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p (nucleus sampling)")
            num_sequences = gr.Slider(1, 5, value=1, step=1, label="Number of Sequences")
            voice_dropdown = gr.Dropdown(choices=get_available_voices(), label="Select Voice", value=get_available_voices()[0] if get_available_voices() else None)
            auto_play_checkbox = gr.Checkbox(label="Auto-play Voice", value=False)
            generate_button = gr.Button("Generate Text and Speech")

    generate_button.click(
        generate_and_speak,
        inputs=[model_dropdown, input_text, max_length, temperature, top_k, top_p, num_sequences, image_input, document_input, voice_dropdown, auto_play_checkbox],
        outputs=[output_text, audio_output]
    )

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
