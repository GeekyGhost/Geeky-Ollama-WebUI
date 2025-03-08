# Geeky Ollama WebUI

## Overview

Geeky Ollama WebUI is a sophisticated web interface for Ollama AI models, combining powerful functionality with an intuitive user experience. This application serves as a comprehensive environment for both everyday users and developers to interact with large language models through a feature-rich Gradio interface. The project emphasizes versatility, offering capabilities ranging from casual conversation to advanced coding assistance and visual analysis.

<img width="1238" alt="Screenshot 2024-09-21 012213" src="https://github.com/user-attachments/assets/97711267-675a-4fbf-b2b2-168be432a549">
<img width="1208" alt="Screenshot 2024-09-21 012148" src="https://github.com/user-attachments/assets/8e96f130-df83-41a9-a9b5-4838b9bc2ec0">
<img width="1243" alt="Screenshot 2024-09-21 011723" src="https://github.com/user-attachments/assets/043861e2-4b68-4e34-b087-f89624860f44">
<img width="1219" alt="Screenshot 2024-09-21 011740" src="https://github.com/user-attachments/assets/084843f9-7345-4c7c-9398-067de1816034">

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Interface Overview](#interface-overview)
5. [Advanced Features](#advanced-features)
6. [Technical Details](#technical-details)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)

## Features

### Core Capabilities
- **Versatile Chat Interface**: Engage in natural conversations with AI models
- **Code Generation**: Create Python code from natural language descriptions
- **Document Analysis**: Extract and analyze text from PDF, DOCX, and TXT files
- **Image Analysis**: Process and interpret images with multimodal AI models
- **Voice Output**: Convert AI responses to speech with customizable voices

### Advanced Tools
- **Retrieval-Augmented Generation (RAG)**: Enhance responses with document-based knowledge
- **Function Calling**: Execute specific functions through natural language requests
- **Structured Output**: Generate responses in JSON format for programmatic processing
- **Code Execution & Linting**: Run and analyze Python code within the interface
- **Multiple Session Management**: Create and switch between different conversation contexts

### Model Management
- **Integrated Model Controls**: Download, delete, and manage Ollama models directly from the UI
- **Modelfile Editing**: Create and modify Ollama Modelfiles through a dedicated interface
- **Parameter Customization**: Fine-tune generation parameters for optimal results

## Installation

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running on your system ([Ollama installation guide](https://ollama.ai/download))

### Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/GeekyGhost/Geeky-Ollama-WebUI.git
   cd Geeky-Ollama-WebUI
   ```

2. **Set up a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**:
   ```bash
   python app.py
   ```

5. **Access the interface**:
   Open your web browser and navigate to http://127.0.0.1:7860

## Usage

### Getting Started
1. **Select a model** from the dropdown menu (requires Ollama to be running with models installed)
2. **Choose a mode** (Chat, Coding, Image Analysis, etc.) based on your task
3. **Type your prompt** in the input field and click "Send"
4. **View the response** in the conversation display

### Working with Documents and Images
The interface provides dedicated upload sections for both documents and images:

1. **Document Upload**:
   - Supports PDF, DOCX, and TXT formats
   - Provides context for AI responses
   - Displays file names and types for easy reference

2. **Image Upload**:
   - Compatible with common image formats
   - Enables visual analysis with multimodal models
   - Supports upload from device, clipboard, or webcam

### Managing Sessions
- Create new sessions to start fresh conversations
- Save and name sessions for future reference
- Switch between different sessions to maintain context separation

## Interface Overview

### Main Tabs
1. **Chat**: Primary interface for conversation and content generation
2. **Code**: Dedicated space for code generation and management
3. **Document QA**: Question-answering based on uploaded documents
4. **Models**: Interface for managing Ollama models
5. **Settings**: Configuration options for the application

### Chat Interface Components
- **Conversation Display**: Shows the ongoing conversation with message history
- **Input Area**: Text field for entering prompts with microphone option
- **Model Selection**: Dropdown to choose the active AI model
- **Mode Selection**: Radio buttons for different interaction types
- **Quick Settings**: Expandable panel for adjusting generation parameters
- **Upload Section**: Area for document and image uploads

## Advanced Features

### RAG (Retrieval-Augmented Generation)
The Document QA tab implements RAG technology to:
- Process and index uploaded documents
- Retrieve relevant information based on queries
- Generate responses grounded in document content

### Function Calling
Enable the function calling feature to:
- Perform specific tasks through natural language
- Execute built-in utilities like calculations or information retrieval
- Get structured results from unstructured requests

### Code Generation and Management
The Code tab provides specialized features:
- Generate Python code based on requirements
- Navigate through code history
- Execute code within a safe environment
- Check code for syntax errors
- Download generated code for external use

## Technical Details

### Architecture
Geeky Ollama WebUI is built on several key technologies:
- **Gradio**: Provides the responsive web interface
- **Ollama API**: Connects to Ollama models for AI capabilities
- **LangChain**: Powers the RAG system for document analysis
- **PyPDF2 & python-docx**: Handle document parsing
- **pyttsx3**: Enables text-to-speech functionality

### File Structure
- `app.py`: Main application file containing the Gradio interface
- `requirements.txt`: Lists all necessary Python dependencies
- `run.bat`: Windows batch file for easy startup

### Model Compatibility
This interface works with all Ollama models, with special features for:
- General text models (chat, completion)
- Code-specific models for programming tasks
- Multimodal models supporting image analysis
- Embedding models for RAG capabilities

## Troubleshooting

### Common Issues
- **"No models found" error**: Ensure Ollama is running and has models installed
- **Image analysis not working**: Verify you're using a multimodal model (e.g., llama3-vision)
- **Voice generation issues**: Check pyttsx3 installation and system audio configuration
- **Document processing errors**: Confirm document format compatibility and file integrity

### Performance Optimization
- Use smaller models for faster responses
- Adjust max length parameter for shorter generations
- Reduce temperature for more deterministic outputs
- Consider hardware limitations when processing large documents or images

## Contributing

Contributions to Geeky Ollama WebUI are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created by Willie G. (Geeky Ghost)

*Powered by Ollama and built with Gradio*
