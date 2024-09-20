# Geeky Ollama WebUI

## Overview

Geeky Ollama WebUI is a sophisticated AI-assisted coding and chat interface built with Gradio and powered by the Ollama AI model. This project aims to explore the capabilities of large language models in a user-friendly web interface, providing a versatile platform for both casual users and developers. It evolved from an earlier GPT-2 UI concept and continues to expand its features and capabilities.

![Geeky Ollama WebUI v2 Screenshot 1](https://github.com/GeekyGhost/Geeky-Ollama-WebUI/assets/111990299/6e42de3e-7c78-4c4e-8240-d54610c6b36e)
![Geeky Ollama WebUI v2 Screenshot 2](https://github.com/GeekyGhost/Geeky-Ollama-WebUI/assets/111990299/b9ac39c6-f3bb-410c-9529-2134ff80e41e)

## Table of Contents

1. [For Users](#for-users)
   - [Features](#features)
   - [Setup](#setup)
   - [Usage Guide](#usage-guide)
   - [Troubleshooting](#troubleshooting)
2. [For Developers](#for-developers)
   - [Architecture Overview](#architecture-overview)
   - [Key Components](#key-components)
   - [Implementation Details](#implementation-details)
   - [Extending the Project](#extending-the-project)
   - [Known Limitations](#known-limitations)
3. [Contributing](#contributing)
4. [License](#license)
5. [Contact](#contact)

## For Users

### Features

1. **Intelligent Chat Interface**: Engage in dynamic conversations with the AI on a wide range of topics.
2. **Advanced Code Generation**: Generate Python code based on natural language requests.
3. **Code Continuation and Refactoring**: Seamlessly continue writing code or refactor existing code with AI assistance.
4. **Voice Synthesis**: Convert text responses to speech for an immersive audio experience.
5. **Document Analysis**: Upload and analyze documents (PDF, DOCX, TXT) to provide context for AI responses.
6. **Image Analysis**: Analyze and discuss uploaded images with compatible AI models.
7. **Comprehensive Model Management**: Download, delete, and manage different AI models directly from the interface.
8. **Code Execution and Linting**: Execute Python code in a safe environment and perform basic code analysis.
9. **Library Explorer**: Discover and import Python libraries within the interface.
10. **RAG (Retrieval-Augmented Generation)**: Enhance AI responses with relevant information retrieval from uploaded documents.
11. **Multiple Chat Sessions**: Efficiently manage and switch between different chat sessions.
12. **Customizable AI Parameters**: Fine-tune generation parameters for optimal outputs.

### Setup

1. Ensure you have Python 3.8+ installed on your system.
2. Clone the repository:
   ```
   git clone https://github.com/GeekyGhost/Geeky-Ollama-WebUI.git
   ```
3. Navigate to the project directory:
   ```
   cd Geeky-Ollama-WebUI
   ```
4. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Install and set up Ollama on your system following the instructions at [Ollama's official website](https://ollama.ai/).

### Usage Guide

1. Start the Ollama service on your system.
2. Run the main script:
   ```
   python geeky-Web-ui-main.py
   ```
3. Open your web browser and navigate to the local address provided by Gradio (typically `http://127.0.0.1:7860`).
4. Use the interface to:
   - Chat with the AI by typing in the input box and clicking "Generate"
   - Switch between Chat and Coding modes using the radio buttons
   - Upload documents or images for analysis
   - Adjust AI generation parameters using the sliders
   - Manage models in the "Model Management" tab
   - Explore and import libraries in the "Library Explorer" tab

### Troubleshooting

- If you encounter model loading issues, ensure Ollama is running and up-to-date.
- For voice generation problems, check that you have the necessary system libraries for pyttsx3.
- If the interface doesn't load, verify that all dependencies are correctly installed and that there are no conflicting versions.

## For Developers

### Architecture Overview

Geeky Ollama WebUI is built on a modular architecture that integrates various components:

1. **Gradio Frontend**: Provides the web interface and handles user interactions.
2. **Ollama Client**: Manages communication with the Ollama API for AI model interactions.
3. **Document Processor**: Handles parsing and text extraction from various file formats.
4. **Code Analyzer**: Utilizes the Abstract Syntax Tree (AST) for code analysis and manipulation.
5. **Voice Synthesizer**: Converts text to speech using pyttsx3.
6. **RAG System**: Implements Retrieval-Augmented Generation for enhanced responses.

The application follows a client-server model, with Gradio serving as the frontend and the Python backend handling the core logic and AI interactions.

### Key Components

1. **ollama.Client**: The main interface for interacting with Ollama models.
2. **gradio.Blocks**: Used to create the dynamic web interface.
3. **PdfReader and Document**: For parsing PDF and DOCX files respectively.
4. **pyttsx3**: Handles text-to-speech conversion.
5. **ast**: Used for Python code analysis and manipulation.
6. **langchain**: Implements the RAG system for enhanced information retrieval.

### Implementation Details

#### Ollama Integration
The `generate_text` function is the core of the AI interaction:

```python
def generate_text(model: str, prompt: str, max_length: int, temperature: float, top_k: int, top_p: float,
                  num_sequences: int, image: Optional[str] = None, context: Optional[str] = None) -> str:
    # ... (function implementation)
```

This function prepares the request to the Ollama API, handling various parameters and optional image input.

#### Code Generation and Analysis
Code generation, continuation, and refactoring are handled by separate functions that utilize the Ollama API and AST for code manipulation:

```python
def continue_code_generation(coding_model: str, current_code: str, user_request: str, max_length: int, temperature: float, top_k: int, top_p: float) -> Tuple[gr.update, str]:
    # ... (function implementation)

def refactor_code(coding_model: str, current_code: str, user_request: str, max_length: int, temperature: float, top_k: int, top_p: float) -> Tuple[gr.update, str]:
    # ... (function implementation)
```

#### RAG Implementation
The RAG system uses langchain components to enhance AI responses:

```python
def process_document(file_path: str, question: str) -> str:
    loader = WebBaseLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text"))
    qa_chain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    result = qa_chain.invoke({"query": question})
    return result['result']
```

#### Gradio Interface
The Gradio interface is created dynamically in the `create_interface` function, which sets up all the UI components and their interactions:

```python
def create_interface():
    with gr.Blocks(title="Enhanced Ollama Text Generation") as iface:
        # ... (interface components and layout)
    return iface
```

### Extending the Project

Developers can extend the project in several ways:

1. **Adding New Models**: Implement support for additional Ollama models or integrate other AI services.
2. **Enhancing RAG**: Improve the RAG system by implementing more sophisticated retrieval methods or document processing techniques.
3. **Expanding Code Analysis**: Implement more advanced code analysis and refactoring techniques using tools like `astroid` or `rope`.
4. **Improving UI**: Enhance the user interface with additional Gradio components or by integrating a custom frontend.
5. **Adding Collaborative Features**: Implement real-time collaboration features using WebSockets or similar technologies.

### Known Limitations

- The project currently only supports Python code generation and analysis.
- RAG implementation is basic and may not handle very large documents efficiently.
- The code execution environment is restricted for security reasons, limiting some functionalities.

## Contributing

Contributions to Geeky Ollama WebUI are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

Please ensure your code adheres to the project's coding standards and include tests for new functionalities.

## License

[Include your chosen license here]

## Contact

[Your contact information or link to issues page]

---

We welcome stars ⭐, forks, and pull requests! If you find this project interesting or useful, please consider contributing or sharing it with others.
