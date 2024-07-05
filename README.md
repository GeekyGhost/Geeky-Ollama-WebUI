Work in progress. Geeky Ollama Web ui, working on RAG and some other things (RAG Done). Rework of my old GPT 2 UI I never fully released due to how bad the output was at the time lol. 


v2 - geeky-Web-ui-main.py

<img width="875" alt="Screenshot 2024-07-04 045428" src="https://github.com/GeekyGhost/Geeky-Ollama-WebUI/assets/111990299/6e42de3e-7c78-4c4e-8240-d54610c6b36e">
<img width="926" alt="Screenshot 2024-07-04 045510" src="https://github.com/GeekyGhost/Geeky-Ollama-WebUI/assets/111990299/b9ac39c6-f3bb-410c-9529-2134ff80e41e">



v1 - geekyOllana-Web-ui-main.py

https://github.com/GeekyGhost/Geeky-Ollama-WebUI/assets/111990299/4bf7f250-f9ed-41e4-8cb9-4f80a8781a76

For Users:
This script is a powerful and versatile AI-assisted coding and chat interface. It leverages the Ollama AI model to provide various functionalities:

Chat Interface: Users can have conversations with the AI, asking questions or seeking information on various topics.
Code Generation: The AI can generate Python code based on user requests, making it an excellent tool for programmers of all levels.
Code Continuation and Refactoring: The interface allows users to continue writing code from where they left off or refactor existing code with AI assistance.
Voice Generation: It can convert text responses into speech, making it accessible for users who prefer audio output.
Document Analysis: Users can upload documents (PDF, DOCX, TXT) for the AI to use as context in its responses.
Image Analysis: For compatible models, users can upload images for the AI to analyze and discuss.
Model Management: Users can download, delete, and manage different AI models directly from the interface.
Code Execution and Linting: The interface allows users to execute Python code and check it for potential issues.
Library Explorer: Users can explore and import Python libraries directly from the interface.

Key features include the ability to switch between chat and coding modes, adjust AI generation parameters, manage multiple chat sessions, and customize voice output.
For Developers:
At a high level, this script creates a Gradio-based web interface that interacts with the Ollama API to provide AI-assisted functionalities. Here's a breakdown of the main components:

Imports and Setup:

The script uses various libraries like gradio, ollama, PyPDF2, pyttsx3, and others for different functionalities.
It sets up logging and defines global variables for managing chat history and sessions.


Ollama Client:

An Ollama client is created to interact with the Ollama API.


Utility Functions:

Functions like get_available_models(), extract_text_from_document(), text_to_speech() handle various tasks.


Core Functionality:

generate_text(): Handles the main interaction with the Ollama API for text generation.
generate_with_context(): Manages the generation of responses in both chat and coding modes.
continue_code_generation() and refactor_code(): Handle code continuation and refactoring.


Interface Management:

Functions like new_session(), load_session(), delete_session() manage chat sessions.
download_model(), delete_model(), load_modelfile(), save_modelfile() handle model management.


Code Analysis:

execute_code(): Runs Python code in a restricted environment.
lint_code(): Performs basic code analysis.


Gradio Interface:

create_interface(): Sets up the Gradio web interface with various components like dropdowns, buttons, and text areas.


Main Execution:

The script creates and launches the Gradio interface when run as the main program.



Novel Approaches:

Integration of multiple AI functionalities (chat, code generation, voice synthesis) in a single interface.
Use of AST (Abstract Syntax Tree) for code analysis and continuation.
Implementation of a restricted Python execution environment for safe code running.
Dynamic model management directly from the interface.

The script demonstrates advanced use of the Gradio library, showcasing how to create a complex, multi-functional interface. It also shows effective integration with external APIs (Ollama) and handling of various data types (text, code, audio, images).
