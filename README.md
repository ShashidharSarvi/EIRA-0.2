# 🧠 EIRA‑0.2: Multimodal Medical QA with LLaMA‑2 & BLIP

> Bridging medical imaging and natural language for clinical question answering.

EIRA‑0.2 is a multimodal deep learning model that performs free-form question answering over medical images and related text. It fuses a visual encoder (BLIP) and a language model (LLaMA-2) to support context-aware, image-text-to-text inference—ideal for clinical decision support and educational tools.

---

## 📦 Features

- 🔬 Medical image + text → free-form medical answer  
- 🤝 Built on `LLaMA-2` and `BLIP`  
- 🏥 Tuned on radiographs, histology slides, and clinical notes  
- 🧪 Evaluated on expert-curated test sets  
- 🚀 Hugging Face-hosted weights (no Git LFS needed)  

---

## 🖼️ Example

```python
from transformers import pipeline

# Load the multimodal QA pipeline
eira = pipeline(
    task="image-text-to-text",
    model="bockhealthbharath/Eira-0.2",
    device=0  # or -1 for CPU
)

# Inference
answer = eira({
    "image": "chest_xray.png",
    "text": "What abnormality is visible in the left lung?"
})

print("Answer:", answer[0]["generated_text"])
