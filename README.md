# ai_chatbot
Create a "models" folder, insert your quantized model there
- HuggingFace user with a lot of said models: https://huggingface.co/TheBloke

To run program:
1. Install dependencies from requirements.txt (pip install -r requirements.txt)
2. Run this command from terminal to allow LLamaCPP to work: CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
