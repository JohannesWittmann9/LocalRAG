from llama_cpp import Llama

class Generator:
    model_path = "./models/generation/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"

    def __init__(self, use_gpu:bool):
        self.use_gpu = use_gpu

        n_gpu_layers = -1 if use_gpu else 0

        self.model = Llama(
            model_path=self.model_path,  # path to GGUF file
            n_ctx=5000,  # The max sequence length to use - note that longer sequence lengths require much more resources
            n_gpu_layers=n_gpu_layers, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
            verbose=True
            )

    def query(self, query: str, ctx: str, debug = False):
        if debug: print("Starting Generation...\n")

        user_prompt = "Es tut uns Leid, wir konnten keinen passenden Kontext finden. Bitte wenden Sie sich an einen Berater oder versuchen Sie die Frage genauer zu formulieren."

        if ctx is not None:
            user_prompt = f"""
Nutze den Kontext um die Frage zu beantworten:
{ctx}

Frage:
{query}
"""

        if debug: print(f"User prompt: {user_prompt}\n")

        # Prompt creation
        system_message = """Dies ist eine Unterhaltung zwischen einem intelligenten, hilfsbereitem KI-Assistenten und einem Nutzer.
                                Der Assistent gibt ausführliche, hilfreiche und ehrliche Antworten."""

        prompt = f"""[INST]{system_message} {user_prompt} [/INST]"""
        

        out = self.model(prompt, max_tokens= 500, stop=["[INST]","[/INST]"], echo=True, stream=True)

        return out