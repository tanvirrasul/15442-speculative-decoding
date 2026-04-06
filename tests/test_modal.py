import modal

app = modal.App("modal-test")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers", "torch", "accelerate", "sentencepiece"
)

@app.function(image=image, gpu="A100", timeout=300, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_llama():
    import os
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        token=os.environ["HF_TOKEN"]
    )
    print("Llama-2 tokenizer loaded successfully!")
    return "ok"

@app.local_entrypoint()
def main():
    result = test_llama.remote()
    print("Modal + HuggingFace working:", result)
