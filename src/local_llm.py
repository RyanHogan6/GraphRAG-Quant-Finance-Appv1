import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class LocalLLM:
    def __init__(self, base_model_id="meta-llama/Llama-3.2-1B-Instruct", 
                 adapter_path="./llama-3.2-1b-aql-lora-final"):
        """Load your fine-tuned model"""
        print("Loading fine-tuned model...")
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
    
    def generate(self, prompt, max_tokens=512, temperature=0.1):
        """Generate response from your model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

# Singleton instance
_llm_instance = None

def get_local_llm():
    """Get or create LLM instance (loads once, reuses)"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM()
    return _llm_instance
