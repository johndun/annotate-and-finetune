from pathlib import Path
from typing import Annotated
import typer
from typer import Option
from transformers import AutoModel, AutoTokenizer

def download_model(
    output_path: Annotated[str, Option(help="Directory to save the model")] = "models/roberta-base",
    model_name: Annotated[str, Option(help="Hugging Face model identifier")] = "FacebookAI/roberta-base",
):
    """Download a Hugging Face model and tokenizer to the specified directory."""
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model {model_name} to {output_path}")
    
    # Download and save model
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Download complete!")

if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(download_model)
    app()
