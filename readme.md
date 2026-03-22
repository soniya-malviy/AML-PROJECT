# Automated EdTech Grading Assistant

## 1. Project Title & Overview  
**Project Name:** Automated EdTech Grading Assistant  

**Description:**  
An AI-powered end-to-end grading system that evaluates handwritten student answer sheets by combining **handwritten text recognition (HTR)** with **semantic answer evaluation**. The system integrates image preprocessing, OCR pipelines, and NLP-based similarity modeling to predict grading outcomes (Pass/Fail) with high accuracy and consistency.

**Problem Statement:**  
To accurately predict binary grading outcomes (Pass/Fail) by modeling the interaction between:
- **Unstructured image data** (scanned handwritten answer sheets)  
- **Semantic textual similarity** (student answers vs reference answers)

---

## 2. Motivation  

- **Educational Impact:**  
  Manual grading is slow, inconsistent, and difficult to scale. This system automates evaluation while maintaining fairness and consistency.

- **Why Image + Text Fusion:**  
  - Image data determines **OCR reliability** (clarity, skew, noise).  
  - Text data determines **conceptual correctness** (semantic similarity).  
  Combining both creates a robust representation of true student understanding.

---

## 3. Dataset  

###  Sources:
- IAM Handwriting Database – handwritten text recognition benchmark  
- Mohler Short Answer Grading Dataset – semantic grading dataset  

###  Features:
- **Image Features:** Ink coverage, skew, noise, aspect ratio  
- **Text Features:** Student answers, reference answers  
- **Tabular Features:** Word count, length ratio, question difficulty  

###  Target Variable:
- **Pass / Fail** (based on grading threshold)

---

## 4. Approach: End-to-End Pipeline  

###  Phase 1: Exploratory Data Analysis  
- Analyzed class imbalance (2.4:1 ratio)  
- Identified TF-IDF overlap zone (0.20–0.50)  
- Detected OCR failure conditions (low ink, high skew)  
- Prevented target leakage  

---

###  Phase 2: Feature Engineering  

#### Image Processing Pipeline:
- Denoising (Bilateral Filter)  
- Deskewing (Hough Transform)  
- Adaptive Binarization (Sauvola)  
- Morphological Cleaning  

#### Text Processing:
- TF-IDF + SVD (LSA)  
- Semantic similarity approximation (SBERT-like embeddings)  

---

###  Engineered Features (Core Innovation)

1. **Similarity Margin** – separates correct vs incorrect overlap  
2. **Length-Decoupled Similarity** – removes word-count bias  
3. **MATTR** – length-invariant lexical diversity  
4. **Question Difficulty Residual** – normalizes grading fairness  
5. **Semantic Confidence** – captures paraphrasing beyond TF-IDF  
6. **OCR Confidence Weight** – penalizes poor-quality scans  

---

###  Phase 3: Modeling & Ablation Study  

| Model Type | Description |
|----------|------------|
| **Baseline** | TF-IDF cosine threshold |
| **Tabular Model** | Logistic Regression on structured features |
| **Semantic Model** | Similarity-based classifier |
| **Domain-Modified** | Class-weighted + threshold tuning |
| **Hybrid Composite** | Full feature fusion model |

---

## 5. Results & Benchmarks  

###  Final Performance (Best Model: Hybrid Composite)

| Metric | Score |
|------|------|
| **F1-Score** | **0.8156** |
| **AUC-ROC** | **0.9043** |
| **Brier Score** | 0.159 |

###  Key Insight:
- Tabular + semantic + OCR features together outperform individual modalities.
- Hybrid fusion provides the most reliable grading predictions.

---

## 6. Project Structure 

```
edtech-grading-assistant/
│
├── data/ # Raw and processed datasets
├── notebooks/ # EDA, feature engineering, modeling
├── models/ # Saved trained models
├── reports/ # Evaluation metrics & analysis
├── docs/ # Documentation
├── requirements.txt # Dependencies
└── README.md # Project overview

```


---

## 7. Installation & Setup  

```bash
# Clone repository
git clone <repo_url>
cd edtech-grading-assistant

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## 8. How to Run

   EDA:
    Run notebooks/01_eda.ipynb
    Feature Engineering:
    Run notebooks/02_feature_engineering.ipynb
    Model Training:
    Run notebooks/03_model_training.ipynb


## 9. Development Methodology

 Modular pipeline design
 Strict separation of raw vs processed data
 Reproducibility (seed = 42)
 Clean version control practices
 Ablation-driven validation
 
## 10. Failure Analysis

 ### False Positives:
  Keyword-heavy but incorrect answers
  Verbose irrelevant responses
  
 ### False Negatives:
  Short but correct answers
  Correct paraphrases (TF-IDF limitation)

## 11. Limitations & Future Work

   Limitations:
    TF-IDF fails on paraphrases (semantic ceiling)
    Distribution shift (IAM → real exam images)
    Domain-specific bias
    
   Future Improvements:
    Fine-tuned SBERT embeddings
    Integration with LLMs for zero-shot grading
    Real-time OCR pipeline optimization
    Multi-subject generalization
    
## 12. Contributors
Soniya Malviya (230159)
Aryan Soni (230049)


## 13. Conclusion

This project demonstrates that combining image quality signals, semantic similarity, and engineered grading features produces a highly reliable automated grading system.

The Hybrid Composite model establishes a strong baseline for AI-driven educational assessment systems, paving the way for future LLM-enhanced grading solutions.


---

