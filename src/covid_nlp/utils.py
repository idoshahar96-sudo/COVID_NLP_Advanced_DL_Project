import logging, random, numpy as np

try:
    import torch
except Exception:
    torch = None

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    if torch:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

def hello():
    return "covid_nlp package is importable 🎯"
