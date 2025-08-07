
---

# English to Hindi Translation using Transformer from Scratch

This project implements a Transformer-based sequence-to-sequence model for **English to Hindi** neural machine translation (NMT) using PyTorch. Inspired by the original Transformer architecture (Vaswani et al., 2017), the model is trained from scratch on a parallel English-Hindi dataset.

---

##  Model Overview

The model is built using the vanilla Transformer architecture with:

* Multi-head Self Attention
* Positional Encoding
* Encoder-Decoder architecture
* Masked Attention
* Greedy & Beam Search decoding

---

##  Folder Structure

```
.
├── config.py              # Configuration & hyperparameters
├── dataset.py             # Data loading & tokenization
├── model.py               # Transformer model definition
├── train.py               # Training loop using GPU
├── train_wb.py            # Optional: Train with Weights & Biases logging
├── translate.py           # Translation inference using trained model
├── Inference.ipynb        # Inference notebook
├── Local_Train.ipynb      # Notebook version of training loop
├── Beam_Search.ipynb      # Beam search implementation
├── attention_visual.ipynb # Visualize attention weights
├── conda.txt              # Environment dependencies
├── requirements.txt       # Python dependencies
└── .gitignore
```

---

##  Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/english-hindi-transformer.git
cd english-hindi-transformer
```

2. Create the conda environment:

```bash
conda create --name en2hi_transformer --file conda.txt
conda activate en2hi_transformer
```

3. Or use pip:

```bash
pip install -r requirements.txt
```

---

##  Dataset

This project uses the **IIT Bombay English-Hindi Parallel Corpus**:

* **Download link:** [https://www.cfilt.iitb.ac.in/iitb\_parallel/](https://www.cfilt.iitb.ac.in/iitb_parallel/)
* Format: `.txt` files with parallel English-Hindi sentences.
* Cleaned and preprocessed in `dataset.py`.

---

##  Training

### Option 1: Script

```bash
python train.py
```

### Option 2: Jupyter Notebook

Use `Local_Train.ipynb` to train interactively.

---

##  Inference

Translate an English sentence using:

```bash
python translate.py --sentence "I love my country"
```

Or open `Inference.ipynb` for testing multiple examples.

---

##  Attention Visualization

Visualize attention using:

```bash
# Inside the notebook
attention_visual.ipynb
```

---

##  Beam Search Decoding

The `Beam_Search.ipynb` notebook provides an improved translation decoding mechanism over greedy decoding.

---

## ⚙ Configuration

You can modify all hyperparameters and path settings in `config.py`, including:

* `max_len`
* `d_model`
* `num_layers`
* `batch_size`
* `device`
* `dataset paths`

---

##  Results

| Metric     | Score                                                                                |
| ---------- | ------------------------------------------------------------------------------------ |
| BLEU Score | *To be evaluated*                                                                    |
| Accuracy   | *Qualitative* — human-verified translation quality is acceptable for basic sentences |

---

##  Tech Stack

* Python 3.x
* PyTorch
* Jupyter
* SentencePiece (tokenizer)
* Matplotlib (attention plots)

---

##  Citation

Based on the architecture introduced in:

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). *Attention is all you need*. Advances in neural information processing systems, 30.

---

##  Contact

For queries or collaboration:

* Gaurav Kumar — [LinkedIn](https://www.linkedin.com)

---

* Example `.txt` English-Hindi pairs to include in the repo for quick testing
* Or badge support (like PyPI, GitHub Actions, etc.)

Would you like this saved as a `README.md` file?
