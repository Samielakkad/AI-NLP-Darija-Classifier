"""
jakma-darija-classifier · Pass-1 inference handler

Wraps a base xlm-roberta-base model with the jak.ma trade × city × confidence
heads. Designed for Hugging Face Inference Endpoints + Spaces.
"""
from typing import Dict, Any, List
import json
import os

# Base model — published weights are a thin policy + tokenizer extension.
# Full LoRA-tuned weights for the classification heads are released alongside.
BASE_MODEL = "xlm-roberta-base"

TRADES = [
    "plumber", "electrician", "tiler", "painter", "carpenter", "mason",
    "mechanic", "electronics", "ac_technician", "gardener", "cleaner", "mover",
]
CITIES = [
    "Casablanca", "Rabat", "Sale", "Tangier", "Marrakesh", "Agadir", "Fes",
    "Meknes", "Oujda", "Kenitra", "Tetouan", "Nador", "Beni Mellal",
    "El Jadida", "Mohammedia",
]
CONFIDENCE = ["low", "medium", "high"]


class EndpointHandler:
    """
    Hugging Face Inference Endpoints / Spaces handler.

    Loads the base model + classification heads, returns structured
    {trade, city, confidence} predictions.
    """

    def __init__(self, path: str = ""):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path or BASE_MODEL)
        self.base = AutoModel.from_pretrained(path or BASE_MODEL).to(self.device).eval()

        # Classification heads (loaded from heads.pt if present, else random init).
        heads_path = os.path.join(path, "heads.pt") if path else None
        if heads_path and os.path.exists(heads_path):
            self.heads = torch.load(heads_path, map_location=self.device)
        else:
            hidden = self.base.config.hidden_size
            self.heads = {
                "trade": torch.nn.Linear(hidden, len(TRADES)).to(self.device),
                "city": torch.nn.Linear(hidden, len(CITIES)).to(self.device),
                "confidence": torch.nn.Linear(hidden, len(CONFIDENCE)).to(self.device),
            }

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Single inference call.

        Input format:
            {"inputs": "بغيت plombier f Casa daba"}

        Output format:
            [{
                "trade": "plumber",
                "city": "Casablanca",
                "confidence": "high",
                "scores": {
                    "trade": 0.94, "city": 0.91, "confidence": 0.89
                }
            }]
        """
        import torch

        text = data.get("inputs", "")
        if isinstance(text, list):
            text = text[0] if text else ""

        toks = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128,
        ).to(self.device)

        with torch.no_grad():
            out = self.base(**toks)
            pooled = out.last_hidden_state[:, 0]  # [CLS]

            trade_logits = self.heads["trade"](pooled)
            city_logits = self.heads["city"](pooled)
            conf_logits = self.heads["confidence"](pooled)

            trade_probs = torch.softmax(trade_logits, dim=-1).squeeze()
            city_probs = torch.softmax(city_logits, dim=-1).squeeze()
            conf_probs = torch.softmax(conf_logits, dim=-1).squeeze()

            trade_idx = int(trade_probs.argmax())
            city_idx = int(city_probs.argmax())
            conf_idx = int(conf_probs.argmax())

        return [{
            "trade": TRADES[trade_idx],
            "city": CITIES[city_idx],
            "confidence": CONFIDENCE[conf_idx],
            "scores": {
                "trade": float(trade_probs[trade_idx]),
                "city": float(city_probs[city_idx]),
                "confidence": float(conf_probs[conf_idx]),
            },
        }]
