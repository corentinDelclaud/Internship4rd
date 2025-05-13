For the models, let's use the following:

Embedding model: hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
Language model: hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
And for the dataset, we will use a simple list of facts about cat. Each fact will be considered as a chunk in the indexing phrase.


Download ollama's models:

After installed, open a terminal and run the following command to download the required models:

ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

If you see the following output, it means the models are successfully downloaded:

pulling manifest
...
verifying sha256 digest
writing manifest
success