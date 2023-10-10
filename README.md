# ai_chatbot
Create a "models" folder, insert your quantized model there
- HuggingFace user with a lot of said models: https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF
    - Scroll down to "vicuna-7b-v1.5.Q4_K_M.gguf" and click it (you can use any other model shown, but I used this)
    - Press "download"

- Inside qa.py, change the directory of where you put your model in the "llama_model_path = <model_directory_here>" (line 20)

To run program:
1. Install dependencies from requirements.txt (pip install -r requirements.txt)
2. Run this command from terminal to allow LLamaCPP to work: CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
    - Note: The above command is for UNIX based systems (Linux, Mac). If you are on Windows, you need to download GitBash terminal and execute there. In VSCode, when you open the terminal, switch it to Gitbash (dropdown arrow)
3. Finally, to run program: chainlit run qa.py
