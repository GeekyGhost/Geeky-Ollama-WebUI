Work in progress. Geeky Ollama Web ui, working on RAG, most else works fine. Rework of my old GPT 2 UI I never fully released due to how bad the output was at the time lol. 


v2 - geeky-Web-ui-main.py

<img width="875" alt="Screenshot 2024-07-04 045428" src="https://github.com/GeekyGhost/Geeky-Ollama-WebUI/assets/111990299/6e42de3e-7c78-4c4e-8240-d54610c6b36e">
<img width="926" alt="Screenshot 2024-07-04 045510" src="https://github.com/GeekyGhost/Geeky-Ollama-WebUI/assets/111990299/b9ac39c6-f3bb-410c-9529-2134ff80e41e">



v1 - geekyOllana-Web-ui-main.py

https://github.com/GeekyGhost/Geeky-Ollama-WebUI/assets/111990299/4bf7f250-f9ed-41e4-8cb9-4f80a8781a76

Overview:
This code implements an "Enhanced Ollama Text Generation" interface using Gradio. It's designed to interact with Ollama, a local large language model (LLM) server, providing a user-friendly web interface for text generation, coding assistance, and various other AI-powered features.
Key Components and Functionality:

Ollama Integration:

Connects to a locally running Ollama server (http://localhost:11434/api).
Supports multiple Ollama models for text generation and coding tasks.


User Interface:

Built with Gradio, offering a web-based interface accessible through a browser.
Includes separate areas for chat interaction and code output.


Text Generation:

Users can input prompts and generate responses from selected Ollama models.
Supports context-aware generation, incorporating chat history and uploaded documents.


Coding Assistance:

Dedicated mode for coding tasks with a separate model selection.
Generates code explanations and implementations based on user requests.


Voice Interaction:

Text-to-speech functionality to vocalize AI responses.
Speech-to-text capability for voice input.


Document Processing:

Supports uploading and processing PDF, DOCX, and TXT files for context.


Image Input:

Allows image uploads for potential image-based prompts or analysis.


Session Management:

Create, load, and delete chat sessions.


Model Management:

Interface to download, delete, and manage Ollama models.
View and edit model configurations (modelfiles).


Customizable Generation Parameters:

Adjustable settings like max length, temperature, top-k, and top-p for fine-tuning outputs.



How to Use:

Ensure Ollama is installed and running locally.
Install required dependencies using the provided requirements.txt.
Run the script to launch the Gradio interface.
Select models for main conversation and coding tasks.
Input prompts in the text area or use voice input.
Adjust generation parameters as needed.
Toggle between chat and coding modes for different tasks.
Utilize document upload for additional context in conversations.
Manage sessions and models through the provided interface.

Unique Features:

Integration of chat and coding assistance in one interface.
Local LLM usage, ensuring privacy and potentially faster response times.
Flexible model selection for different tasks.
Voice interaction capabilities.
Document and image input for enhanced context.
Customizable AI parameters for fine-tuned outputs.

Work in Progress:
This code is a work in progress, which means:

It may contain bugs or unoptimized sections.
Some features might not be fully implemented or tested.
The user interface and functionality are subject to change and improvement.
Error handling and edge cases might not be fully addressed.
Documentation and code comments may be incomplete.
Performance optimizations are likely needed for larger scale use.

Future improvements could include:

Enhanced error handling and user feedback.
More robust session management and data persistence.
Expanded model management features.
Improved UI/UX design.
Additional integrations with other AI services or tools.
Optimizations for performance and resource usage.

This project showcases an ambitious attempt to create a versatile, user-friendly interface for interacting with local LLMs, combining various AI-powered features in a single application. As it evolves, it has the potential to become a powerful tool for developers, researchers, and AI enthusiasts working with language models.

