from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_qwen_model():
    """Download Qwen3-1.7B model and store it locally"""

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models" / "qwen3-1.7b"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = "Qwen/Qwen3-1.7B"

    print(f"Downloading {model_name} to {models_dir}")

    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(models_dir)

        # Download model
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.save_pretrained(models_dir)

        print(f"Model successfully downloaded to {models_dir}")

        # Create a model info file
        info_file = models_dir / "model_info.txt"
        with open(info_file, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Downloaded from: https://huggingface.co/{model_name}\n")
            f.write(f"Local path: {models_dir}\n")
            f.write("Model type: Causal Language Model\n")
            f.write("Parameters: 1.7B\n")

    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

    return True


if __name__ == "__main__":
    download_qwen_model()
