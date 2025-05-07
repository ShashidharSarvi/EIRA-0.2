 # EIRA-0.2

Multimodal medical QA using LLaMA-2 + BLIP. No need for 245MB of PyTorch or huge infrastructure—this model runs directly with Hugging Face transformers and a GPU. Focus is on image-text to free-form clinical QA. Designed for research, reproducibility, and speed.  
Architecture: BLIP (image encoder) → cross-attn → LLaMA-2 decoder.  
Training data: curated radiographs, histopathology, textbook QA.  
Evaluation on real medical exam-style questions and textbook-derived datasets.  
Runs inference in <2s/question on consumer GPUs.

---

## quick start

Clone the repo and run a demo on your own medical image:

```bash
git clone https://github.com/BockBharath/EIRA-0.2.git
cd EIRA-0.2
pip install -r requirements.txt
```

Run QA:

```python
from transformers import pipeline

eira = pipeline(
  task="image-text-to-text",
  model="bockhealthbharath/Eira-0.2",
  device=0  # or -1 for CPU
)

out = eira({
  "image": "demo/chest_xray.png",
  "text": "What is the most likely diagnosis?"
})
print(out[0]["generated_text"])
```

---

## weights

Model weights hosted at:  
https://huggingface.co/bockhealthbharath/Eira-0.2

No LFS or custom loaders. Fully Hugging Face-compatible.

---

## requirements

- torch >= 2.0  
- transformers >= 4.38  
- accelerate  
- pillow  
- torchvision  

**optional (for training):**  
- deepspeed  
- datasets

---

## datasets

Training data (not publicly released) includes:
- PubMed-derived radiology reports
- MIMIC-CXR (text only)
- WHO textbook-based QA pairs
- Custom histopathology annotations

---

## testing

```bash
python scripts/test_eira.py \
  --image demo/chest_xray.png \
  --question "What is the most likely diagnosis?"
```

---

## training (coming soon)

Training code is under development. Structure:

- `model.py`: BLIP encoder + Q-Former + LLaMA-2 decoder
- `train.py`: Hugging Face Trainer + custom collator
- `configs/`: YAML configs for multiple modalities
- `scripts/`: CLI utilities

---

## license

MIT

---

## acknowledgements

Built with:
- BLIP-2 (Salesforce)
- LLaMA-2 (Meta)
- Hugging Face Transformers
- 🤗 Datasets + Accelerate

---

## goals

EIRA-0.2 is designed to:
- Be reproducible end-to-end
- Support radiology, pathology, and textbook medical QA
- Minimize system dependencies
- Run on consumer GPUs

---

## future

- Multi-language support
- Real-time QA on PACS viewers
- Finetuning for specialties (neuro, derm, etc.)
- Flash Attention integration

---

## forks / ports

- `EIRA.cpp`: C++ port (WIP)
- `EIRA.rs`: Rust port for edge deployment (WIP)
- `EIRA.onnx`: Inference optimization for low-resource devices

---

## community

- Issues → bugs or requests  
- Discussions → usage / tips  
- PRs → welcome!  
- Discord → Coming soon
