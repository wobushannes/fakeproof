# fakeproof

**Local toolkit for AI text generation and detection – no cloud, no bullshit.**

---

## ⚠️ IMPORTANT DISCLAIMER ⚠️

This toolkit is for **research and educational purposes only**. It is designed to demonstrate AI text generation and detection techniques in a controlled, local environment. The user assumes full responsibility for any use of this software, including but not limited to: generating content, training models, and deploying detection systems. The author does not condone or support the use of this toolkit for: disinformation campaigns, academic dishonesty, fraud, harassment, surveillance, or any other unethical or illegal activity. By using this software, you agree that you are solely responsible for complying with all applicable laws and ethical guidelines in your jurisdiction. The author provides this software "as is" and accepts no liability for any damages or consequences arising from its use. You have been warned.

---

## ⚠️ PERFORMANCE & ACCURACY DISCLAIMER ⚠️

The performance of this system – including detection accuracy, false positive/negative rates, and overall reliability – is **entirely dependent on the quality, quantity, and diversity of the training data you provide**. Garbage in, garbage out. If you feed it shit data, it will produce shit results. If you don't have enough real texts, or your fake texts are too similar to real ones, the model will fail. If your data is biased, your model will be biased. This is not a magic bullet. This is a tool. The author makes **no claims, guarantees, or warranties** regarding the accuracy, completeness, or fitness for any particular purpose of the generated models or detection results. The system is provided "as is" and you are solely responsible for validating its performance on your specific use case. Do not rely on this system for any critical decisions or applications where errors could cause harm. Test thoroughly. Validate constantly. And don't come crying to me when it doesn't work perfectly – I told you so.

---

## TL;DR

This is my personal toolbox for generating fake texts via local LLM (LM Studio), augmenting them for more training data, importing real texts into SQLite, training a BERT-based classifier (Fake vs. Real), and testing the trained model with explanations. All GUI-based, all local. No raw data, no models in this repo.

Why? Because I hate cloud dependency. Everything runs locally on my machine. These tools are for me – if you find them useful, great. But don't cry if they break.

---

## Tools Overview

**Tools (`tools/`):**
- `llm.py` – Fake text generator with GUI. Needs LM Studio running locally. Supports multiple prompt styles (politician speeches, social media, technical reports, etc.) and custom topic lists. Saves texts as `fake_XXXX.txt`.
- `aug.py` – Text augmentation using EDA (synonym replacement, random insertion, swap, deletion). Creates multiple variations of each fake text to expand training data.
- `scraper.py` – CSV to SQLite importer with automatic deduplication via MD5 hashing. Imports real texts (e.g., speeches) into `text_corpus.db` with metadata (char count, word count, source file).

**Scripts (`scripts/`):**
- `trainer.py` – Train a ModernBERT-based classifier with GUI. Features checkpoint system (resume training), live progress plots, balanced data loading (fakes + real texts from DB), and automatic model saving with test metrics.
- `tester.py` – Test individual texts with loaded model. Shows fake/real probability and token-level explanation with color-coded word importance (red shades for KI, green for human).

---

## Quick Workflow

1. Import real texts into DB (`scraper.py`)
2. Generate fake texts (`llm.py`) + augment (`aug.py`)
3. Train model (`trainer.py`)
4. Test texts (`tester.py`)

---

## Installation

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

---

**Data & Models:** None provided. No raw data, no pretrained models. Why? Privacy, no copyright headaches, and if you want it, run the tools yourself. The `text_corpus.db` and `fakes/` folder are created by the tools. Save your trained model separately.

---

**FAQ:** 
Can I get access to your data or models? No. Access is restricted. Use the tools to generate your own data and train your own models. 
Can I use it commercially? No. Commercial use is strictly prohibited. No licenses. No exceptions. 
Can I get a live demo? No. Run the tools yourself if you want to see them working. 
Why so restrictive? Because I don't want my work misused for surveillance, disinformation, or other bullshit. I take this seriously – even if others don't. 
Can I collaborate? Yes – if you're from a university, institute, or comparable research environment. Contact me with your profile, institution, and concrete proposal. All other inquiries will be ignored. 
I'm from a law enforcement or security agency. I take legitimate security inquiries seriously. 
Contact me with your credentials and a formal request. 
Can I discuss the research with you? Yes – in academic or policy contexts. Contact me with a clear proposal. 
Can I hire you as a consultant? I offer consulting and research collaboration for institutions with a clear ethical framework. Contact me for details.

---

**Contact:** Serious inquiries only. 
ProtonMail: `blende_32@protonmail.com`. 
Threema: `BA46EWMP`. Before contacting me: 
Provide full name and institution, state your concrete purpose, don't ask for code or access – it will be ignored.

**License:** All rights reserved. © 2026 Johannes Wobus – fakeproof. *"We show what's possible – not how it's done."*
