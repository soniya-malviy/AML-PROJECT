# Phase 1 - Automated EdTech Grading Assistant

## Datasets
- **Mohler Short Answer Grading** (`nkazi/MohlerASAG`) - Real educational dataset for short answer grading

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `eda.ipynb` | Dataset quality & non-obvious patterns |
| `feature_engineering.ipynb` | Novel representations that simplify learning |
| `theoretical_rigor.ipynb` | Assumptions tested with explanations, violations mitigated |
| `model_application_failure_analysis.ipynb` | Customized model with mathematical failure diagnosis |

## Requirements
```
datasets>=2.0
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
scipy>=1.7
scikit-learn>=1.0
nltk>=3.6
```

## Run
```bash
cd phase1
pip install -r requirements.txt
jupyter notebook
```
