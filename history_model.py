from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HistoryModel:
    def __init__(self, base_model: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, do_sample=True, top_p=0.95, temperature=0.8)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
