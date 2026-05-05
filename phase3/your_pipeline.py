# !pip install -q sentence-transformers "chandra-ocr[hf]" bitsandbytes accelerate sympy spacy scikit-learn
# !python -m spacy download en_core_web_trf

import os, getpass

if not os.getenv('HF_TOKEN'):
    os.environ['HF_TOKEN'] = getpass.getpass('Enter your HuggingFace Token: ')
    print('✅ Token set.')
else:
    print('✅ HF_TOKEN already present.')

import gc, os
from pathlib import Path
from PIL import Image

def _load_chandra(hf_token=None, use_4bit=True):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from chandra.model import InferenceManager

    gc.collect()
    torch.cuda.empty_cache()

    model_id = 'datalab-to/chandra-ocr-2'
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    ) if use_4bit else None

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=quant_cfg,
        device_map='auto', trust_remote_code=True, token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=hf_token)

    manager = InferenceManager(method='hf')
    manager.model = model
    manager.processor = processor
    if not hasattr(manager.model, 'processor'):
        manager.model.processor = processor
    return manager

class OCRPipeline:
    def __init__(self, hf_token=None, use_4bit=True):
        self._manager = _load_chandra(hf_token or os.getenv('HF_TOKEN'), use_4bit)

    def extract(self, source):
        image = Image.open(source).convert('RGB') if isinstance(source, (str, Path)) else source.convert('RGB')
        from chandra.model.schema import BatchInputItem
        return self._manager.generate([BatchInputItem(image=image, prompt_type='ocr_layout')])[0].markdown

print('✅ OCR module defined.')

import re, unicodedata

def strip_html(text):
    text = re.sub(r'<math[^>]*>', '', text)
    text = re.sub(r'</math>', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    for entity, char in [('&amp;','&'),('&lt;','<'),('&gt;','>'),('&nbsp;',' '),('&quot;','"')]:
        text = text.replace(entity, char)
    return text

def strip_markdown(text):
    text = re.sub(r'```(?:mermaid|python|java|cpp)?[\s]*\n?', '', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    return text

def clean_for_text(text, keep_math=False):
    """Prepares text for DeBERTa / spaCy.
    
    Args:
        keep_math: If True, keeps LaTeX math content for semantic analysis
    """
    text = strip_html(text)
    
    if not keep_math:
        # Remove LaTeX delimiters but keep the content
        text = re.sub(r'\$\$(.+?)\$\$', r' \1 ', text, flags=re.DOTALL)
        text = re.sub(r'\$(.+?)\$', r' \1 ', text, flags=re.DOTALL)
    
    text = strip_markdown(text)
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return '\n'.join(line.strip() for line in text.split('\n')).strip()

print('✅ Preprocessor ready.')

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def _extract_math_expressions(text):
    """Extract math from both LaTeX blocks AND plain OCR text.
    
    BUG FIX 1: Chandra outputs plain text like 'x^2 + 2x + 1', not LaTeX.
    BUG FIX 2: Fixed regex typo \$$ -> \$\$
    """
    expressions = []
    
    # 1. Extract LaTeX blocks (if they exist)
    blocks_double = re.findall(r"\$\$(.+?)\$\$", text, flags=re.DOTALL)
    expressions.extend([b.strip() for b in blocks_double if b.strip()])
    
    # Remove double-dollar blocks before extracting single-dollar
    text_without_double = re.sub(r"\$\$(.+?)\$\$", " ", text, flags=re.DOTALL)
    blocks_single = re.findall(r"\$([^$]+)\$", text_without_double)
    expressions.extend([b.strip() for b in blocks_single if b.strip()])
    
    # 2. Extract plain math expressions (NEW - handles OCR output)
    # Look for patterns like: x^2 + 2x + 1, (y+1)^2, E = mc^2, etc.
    plain_math_patterns = [
        r'[a-zA-Z]\s*[=]\s*[^\n]{3,50}',  # equations: x = ...
        r'\([a-zA-Z][+\-*/^].*?\)\^\d',   # powers: (x+1)^2
        r'[a-zA-Z]\^\d+\s*[+\-]',         # polynomials: x^2 + ...
        r'\d+\s*[a-zA-Z]\^\d',            # coefficients: 2x^2
    ]
    
    for pattern in plain_math_patterns:
        matches = re.findall(pattern, text)
        expressions.extend([m.strip() for m in matches if m.strip()])
    
    return list(set(expressions))  # Deduplicate

def _parse_to_sympy(expr_str):
    """Convert text/LaTeX to SymPy expression."""
    try:
        # Try direct LaTeX parsing first
        from latex2sympy2 import latex2sympy
        return sympy.simplify(sympy.expand(latex2sympy(expr_str)))
    except:
        pass
    
    try:
        # Try plain text parsing with transformations
        # Convert ^ to ** for SymPy
        expr_str = expr_str.replace('^', '**')
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(expr_str, transformations=transformations)
        return sympy.simplify(sympy.expand(expr))
    except:
        return None

def _canonicalize_variables(student_expr, reference_expr):
    """BUG FIX 3: Rename variables in student answer to match reference.
    
    Example: student has (y+1)^2, reference has (x+1)^2
    After canonicalization: both become (x+1)^2
    """
    if student_expr is None or reference_expr is None:
        return student_expr
    
    # Get free symbols from both
    stu_vars = sorted(student_expr.free_symbols, key=str)
    ref_vars = sorted(reference_expr.free_symbols, key=str)
    
    if len(stu_vars) != len(ref_vars):
        return student_expr  # Different number of variables, can't canonicalize
    
    # Create substitution mapping
    var_map = {s: r for s, r in zip(stu_vars, ref_vars)}
    
    return student_expr.subs(var_map)

def evaluate_math(student_text, reference_text):
    """works with plain OCR text, not just LaTeX."""
    stu_exprs_raw = _extract_math_expressions(student_text)
    ref_exprs_raw = _extract_math_expressions(reference_text)
    
    if not ref_exprs_raw:
        return {"math_score": None, "details": []}
    if not stu_exprs_raw:
        return {"math_score": 0.0, "details": []}
    
    # Parse all expressions
    ref_exprs = [(raw, _parse_to_sympy(raw)) for raw in ref_exprs_raw]
    ref_exprs = [(raw, expr) for raw, expr in ref_exprs if expr is not None]
    
    if not ref_exprs:
        return {"math_score": None, "details": []}
    
    stu_exprs = [(raw, _parse_to_sympy(raw)) for raw in stu_exprs_raw]
    stu_exprs = [(raw, expr) for raw, expr in stu_exprs if expr is not None]
    
    if not stu_exprs:
        return {"math_score": 0.0, "details": []}
    
    total = 0.0
    details = []
    
    # For each reference expression, find best student match
    for r_raw, r_expr in ref_exprs:
        best_score = 0.0
        best_match = None
        
        for s_raw, s_expr in stu_exprs:
            # Try direct comparison
            try:
                diff = sympy.simplify(s_expr - r_expr)
                if diff == 0 or sympy.trigsimp(diff) == 0:
                    best_score = 1.0
                    best_match = s_raw
                    break
            except:
                pass
            
            #  Try with variable canonicalization
            try:
                s_canon = _canonicalize_variables(s_expr, r_expr)
                diff = sympy.simplify(s_canon - r_expr)
                if diff == 0 or sympy.trigsimp(diff) == 0:
                    best_score = 1.0
                    best_match = s_raw
                    break
            except:
                pass
            
            # Handle equations
            if getattr(s_expr, 'is_Relational', False) and getattr(r_expr, 'is_Relational', False):
                try:
                    eq_diff = sympy.simplify(
                        sympy.simplify(s_expr.lhs - s_expr.rhs) - 
                        sympy.simplify(r_expr.lhs - r_expr.rhs)
                    )
                    if eq_diff == 0 or sympy.trigsimp(eq_diff) == 0:
                        best_score = 1.0
                        best_match = s_raw
                        break
                except:
                    pass
            
            # Partial credit for structural similarity
            try:
                s_atoms = s_expr.atoms(sympy.Symbol)
                r_atoms = r_expr.atoms(sympy.Symbol)
                if r_atoms:
                    overlap = len(s_atoms & r_atoms) / len(r_atoms)
                    score = overlap * 0.5
                    if score > best_score:
                        best_score = score
                        best_match = s_raw
            except:
                pass
        
        total += best_score
        details.append({
            "ref": r_raw, 
            "student_match": best_match,
            "score": best_score
        })
    
    return {
        "math_score": round(total / len(details), 4),
        "details": details
    }

print('✅ SymPy Math Engine ready (FIXED: plain text + LaTeX + variable canonicalization).')

import numpy as np

_cross_encoder = None

def _chunk_text(text, max_tokens=450):
    """BUG FIX 5: Split text to avoid 512-token truncation.
    
    DeBERTa has a hard 512 token limit. We chunk at ~450 to be safe.
    """
    # Simple word-based chunking (rough approximation)
    words = text.split()
    chunks = []
    current_chunk = []
    current_count = 0
    
    for word in words:
        # Rough estimate: 1 word ≈ 1.3 tokens
        word_tokens = len(word) // 4 + 1
        if current_count + word_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_count = word_tokens
        else:
            current_chunk.append(word)
            current_count += word_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks if chunks else [text]

def get_nli_entailment_score(student_text, reference_text):
    """ Corrected NLI direction to [student, reference].
     Handles long texts with chunking.
    
    We want: "Does the student's answer contain the reference knowledge?"
    NOT: "Does the reference entail the student's answer?"
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    
    # Chunk both texts to avoid truncation
    stu_chunks = _chunk_text(student_text)
    ref_chunks = _chunk_text(reference_text)
    
    # Compute entailment for each ref chunk against all student chunks
    scores = []
    for ref_chunk in ref_chunks:
        chunk_scores = []
        for stu_chunk in stu_chunks:
            # This asks: "Does student text entail/contain the reference knowledge?"
            logits = _cross_encoder.predict([[stu_chunk, ref_chunk]])[0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            chunk_scores.append(float(probs[1]))  # entailment probability
        
        # Take max score across student chunks (best match)
        scores.append(max(chunk_scores))
    
    # Average across reference chunks
    return sum(scores) / len(scores) if scores else 0.0

print('✅ CrossEncoder Engine ready (FIXED: direction + chunking).')

import spacy
from sentence_transformers import SentenceTransformer, util

_nlp = None
_semantic_model = None

def check_rubric_concepts(student_text, rubric_list, use_semantic=True):
    """ Added semantic matching for rubric concepts.
    
    Now supports:
    - Exact lemma matching (original)
    - Semantic similarity (NEW) - handles paraphrases
    
    Example: "transparent membrane at the front of the eye" matches "cornea"
    """
    global _nlp, _semantic_model
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_trf")
        except:
            _nlp = spacy.load("en_core_web_sm")
    
    if use_semantic and _semantic_model is None:
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    if not rubric_list or not student_text.strip():
        return {"rubric_coverage": None, "concept_scores": {}}
    
    doc = _nlp(student_text)
    total = 0.0
    details = {}
    
    # Split student text into sentences for semantic search
    sentences = [sent.text for sent in doc.sents]
    
    for item in rubric_list:
        concept = item.get("concept", "")
        expected_neg = item.get("requires_negation", False)
        
        # Method 1: Exact lemma matching (original)
        lemma = concept.lower()
        found_exact = False
        negated = False
        
        for token in doc:
            if token.lemma_.lower() == lemma:
                found_exact = True
                if any(child.dep_ == "neg" for child in token.children):
                    negated = True
                else:
                    for anc in token.ancestors:
                        if any(child.dep_ == "neg" for child in anc.children):
                            negated = True
                            break
                break
        
        # Method 2: Semantic similarity (NEW)
        semantic_score = 0.0
        best_match = None
        
        if use_semantic and sentences and _semantic_model:
            concept_embedding = _semantic_model.encode(concept, convert_to_tensor=True)
            sentence_embeddings = _semantic_model.encode(sentences, convert_to_tensor=True)
            
            similarities = util.cos_sim(concept_embedding, sentence_embeddings)[0]
            max_sim = float(similarities.max())
            
            if max_sim > 0.5:  # Threshold for semantic match
                semantic_score = max_sim
                best_match = sentences[similarities.argmax()]
        
        # Combine both methods
        if found_exact and negated == expected_neg:
            score = 1.0
            method = "exact_match"
        elif semantic_score > 0.7:  # High confidence semantic match
            score = semantic_score
            method = "semantic_match"
        elif semantic_score > 0.5:  # Partial semantic match
            score = semantic_score * 0.7
            method = "partial_semantic"
        else:
            score = 0.0
            method = "not_found"
        
        total += score
        details[concept] = {
            "score": round(score, 4),
            "method": method,
            "found_exact": found_exact,
            "semantic_similarity": round(semantic_score, 4) if semantic_score > 0 else None,
            "best_semantic_match": best_match
        }
    
    return {
        "rubric_coverage": round(total / len(rubric_list), 4),
        "concept_scores": details
    }

print('✅ spaCy Engine ready (FIXED: semantic rubric matching).')

def detect_question_boundaries(ocr_text):
    """BUG FIX 6: Detect individual questions in exam OCR.
    
    Returns list of (question_number, question_text) tuples.
    """
    # Common question markers
    patterns = [
        r'^Q\.?\s*(\d+)[:\.]?\s*(.+?)(?=^Q\.?\s*\d+|\Z)',  # Q1: ...
        r'^Question\s+(\d+)[:\.]?\s*(.+?)(?=^Question\s+\d+|\Z)',  # Question 1: ...
        r'^(\d+)[:\.)\.]\s+(.+?)(?=^\d+[:\.)\.]|\Z)',  # 1. or 1) or 1: ...
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, ocr_text, re.MULTILINE | re.DOTALL | re.IGNORECASE))
        if matches:
            questions = []
            for match in matches:
                q_num = int(match.group(1))
                q_text = match.group(2).strip()
                questions.append((q_num, q_text))
            return questions
    
    # No questions detected - treat entire text as one question
    return [(1, ocr_text)]

print('✅ Question boundary detector ready.')

def auto_detect_question_type(text):
    """Automatically detect if question is proof/theory/mixed.
    
    Returns: 'proof', 'theory', or 'mixed'
    """
    text_lower = text.lower()
    
    # Math indicators
    math_indicators = [
        'prove', 'proof', 'derive', 'calculate', 'solve', 'find the value',
        'equation', 'integral', 'derivative', 'theorem', 'lemma',
        '=', '^', '+', '-', '*', '/', '∫', '∑', '∏'
    ]
    
    # Theory indicators
    theory_indicators = [
        'explain', 'describe', 'discuss', 'what is', 'define',
        'why does', 'how does', 'compare', 'contrast', 'analyze'
    ]
    
    math_count = sum(1 for ind in math_indicators if ind in text_lower)
    theory_count = sum(1 for ind in theory_indicators if ind in text_lower)
    
    # Check for mathematical expressions
    has_math_expr = bool(_extract_math_expressions(text))
    
    if has_math_expr or math_count >= 2:
        if theory_count >= 2:
            return 'mixed'
        return 'proof'
    elif theory_count >= 1:
        return 'theory'
    else:
        return 'mixed'  # Default to mixed

print('✅ Auto question-type detector ready.')

def grade_fused(math_res, semantic_score, rubric_res, question_type="mixed", 
                pass_thresh=5.0, custom_weights=None):
    """BUG FIX 10: Now transparently reports actual weights used.
    
    Args:
        custom_weights: Optional dict from training module
    """
    # Default weights
    default_weights = {
        "proof":   {"math": 0.60, "semantic": 0.20, "rubric": 0.20},
        "theory":  {"math": 0.05, "semantic": 0.50, "rubric": 0.45},
        "mixed":   {"math": 0.35, "semantic": 0.35, "rubric": 0.30},
    }
    
    # Use custom weights if provided (from training)
    if custom_weights:
        w = dict(custom_weights)
        original_weights = dict(custom_weights)  # For reporting
    else:
        w = dict(default_weights.get(question_type, default_weights["mixed"]))
        original_weights = dict(w)  # For reporting
    
    math_score = math_res.get("math_score")
    if math_score is None:
        w["semantic"] += w["math"]
        w["math"] = 0.0
        math_score = 0.0
        
    rubric_score = rubric_res.get("rubric_coverage")
    if rubric_score is None:
        w["semantic"] += w["rubric"]
        w["rubric"] = 0.0
        rubric_score = 0.0
    
    final_scale = (
        w["math"] * math_score +
        w["semantic"] * semantic_score +
        w["rubric"] * rubric_score
    )
    
    score_10 = round(final_scale * 10, 1)
    verdict = "Pass" if score_10 >= pass_thresh else "Fail"
    
    return {
        "overall_score": score_10,
        "verdict": verdict,
        "components": {
            "math_sym_score": round(math_score, 4),
            "nli_entailment": round(semantic_score, 4),
            "rubric_coverage": round(rubric_score, 4) if rubric_score is not None else 0.0
        },
        "weights_config": original_weights,  # What was configured
        "weights_actual": w  # What was actually used (FIXED: now transparent)
    }

print('✅ Fusion Logic ready (FIXED: transparent weight reporting).')

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

class GradingTrainer:
    """BUG FIX 9: Train the grading system on human-labeled data.
    
    Three training approaches:
    1. Weight Optimization: Learn optimal fusion weights
    2. Classifier: Train a small model on top of engine scores
    3. Hybrid: Both
    """
    
    def __init__(self):
        self.weight_model = None  # Ridge regression for weights
        self.classifier = None    # Logistic for pass/fail
        self.optimal_weights = None
    
    def collect_training_data(self, pipeline, examples):
        """Extract features from grading engines.
        
        Args:
            pipeline: GradingPipeline instance
            examples: List of dicts with keys:
                - 'image_path': path to student answer
                - 'reference': reference answer
                - 'rubric': rubric rules
                - 'human_score': ground truth score (0-10)
                - 'question_type': 'proof'/'theory'/'mixed'
        
        Returns:
            X: Feature matrix (N x 3) - [math_score, semantic_score, rubric_score]
            y: Ground truth scores (N,)
        """
        X = []
        y = []
        
        print(f"Extracting features from {len(examples)} training examples...")
        
        for i, ex in enumerate(examples):
            print(f"  Processing {i+1}/{len(examples)}...", end='\r')
            
            # Run pipeline to get engine scores
            report = pipeline.grade(
                ex['image_path'],
                ex['reference'],
                ex['rubric'],
                ex.get('question_type', 'mixed'),
                extract_features_only=True  # Don't fuse yet
            )
            
            # Extract raw engine scores
            math_score = report['components']['math_sym_score']
            semantic_score = report['components']['nli_entailment']
            rubric_score = report['components']['rubric_coverage']
            
            X.append([math_score, semantic_score, rubric_score])
            y.append(ex['human_score'])
        
        print("\n✅ Feature extraction complete.")
        return np.array(X), np.array(y)
    
    def train_weights(self, X, y, question_type='mixed'):
        """Approach 1: Learn optimal fusion weights.
        
        Uses Ridge regression with positive constraint.
        """
        print(f"\nTraining optimal weights for question_type='{question_type}'...")
        
        # Ridge regression to learn weights
        # We want: w1*math + w2*semantic + w3*rubric ≈ y/10
        model = Ridge(alpha=0.01, positive=True)
        model.fit(X, y / 10.0)  # Normalize to 0-1 scale
        
        # Extract weights and normalize to sum to 1
        raw_weights = model.coef_
        weight_sum = raw_weights.sum()
        
        if weight_sum > 0:
            optimal_weights = {
                'math': float(raw_weights[0] / weight_sum),
                'semantic': float(raw_weights[1] / weight_sum),
                'rubric': float(raw_weights[2] / weight_sum)
            }
        else:
            # Fallback to equal weights
            optimal_weights = {'math': 0.33, 'semantic': 0.33, 'rubric': 0.34}
        
        # Evaluate
        predictions = model.predict(X) * 10
        mae = np.mean(np.abs(predictions - y))
        
        print(f"  Optimal weights: {optimal_weights}")
        print(f"  Training MAE: {mae:.2f} points")
        
        self.optimal_weights = optimal_weights
        self.weight_model = model
        
        return optimal_weights
    
    def train_classifier(self, X, y, threshold=5.0):
        """Approach 2: Train pass/fail classifier.
        
        Useful for binary grading decisions.
        """
        print(f"\nTraining pass/fail classifier (threshold={threshold})...")
        
        # Binary labels
        y_binary = (y >= threshold).astype(int)
        
        # Logistic regression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y_binary)
        
        # Cross-validation accuracy
        cv_scores = cross_val_score(clf, X, y_binary, cv=min(5, len(X)))
        
        print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.classifier = clf
        return clf
    
    def predict_score(self, math_score, semantic_score, rubric_score, use_classifier=False):
        """Predict score using trained model."""
        X = np.array([[math_score, semantic_score, rubric_score]])
        
        if use_classifier and self.classifier:
            # Binary prediction
            return "Pass" if self.classifier.predict(X)[0] == 1 else "Fail"
        elif self.weight_model:
            # Regression prediction
            return round(float(self.weight_model.predict(X)[0] * 10), 1)
        else:
            raise ValueError("No model trained yet")
    
    def save(self, path):
        """Save trained models."""
        with open(path, 'wb') as f:
            pickle.dump({
                'weight_model': self.weight_model,
                'classifier': self.classifier,
                'optimal_weights': self.optimal_weights
            }, f)
        print(f"✅ Models saved to {path}")
    
    def load(self, path):
        """Load trained models."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weight_model = data['weight_model']
            self.classifier = data['classifier']
            self.optimal_weights = data['optimal_weights']
        print(f"✅ Models loaded from {path}")

print('✅ Training module ready.')

import time
from IPython.display import clear_output

class GradingPipeline:
    
    def __init__(self, use_trained_weights=False):
        print('Initializing Pipeline...')
        self.ocr = OCRPipeline()
        self.trainer = GradingTrainer()
        self.use_trained_weights = use_trained_weights
        print('✅ Loaded Chandra.')
    
    def load_trained_model(self, path):
        """Load trained weights/classifier."""
        self.trainer.load(path)
        self.use_trained_weights = True
    
    def grade(self, image_path, ref_text, rubric_rules, q_type=None, 
              pass_thresh=5.0, extract_features_only=False):
        """Grade a student answer with all fixes applied."""
        t0 = time.time()
        
        print(f'1. OCR Extracting...')
        raw_ocr = self.ocr.extract(image_path)
        
        # Detect question boundaries
        questions = detect_question_boundaries(raw_ocr)
        
        # For now, process first question (extend to multi-question later)
        q_num, q_text = questions[0]
        
        # Auto-detect question type if not provided
        if q_type is None:
            q_type = auto_detect_question_type(q_text)
            print(f'  Auto-detected question type: {q_type}')
        
        print(f'2. Parsing text...')
        #  Keep math in semantic analysis
        stu_clean = clean_for_text(q_text, keep_math=True)
        ref_clean = clean_for_text(ref_text, keep_math=True)
        
        print(f'3. Engine A (SymPy) evaluating...')
        #  math extraction and canonicalization
        math_res = evaluate_math(q_text, ref_text)
        
        print(f'4. Engine B (DeBERTa) evaluating...')
        # direction and chunking
        semantic_score = get_nli_entailment_score(stu_clean, ref_clean)
        
        print(f'5. Engine C (spaCy+Semantic) verifying...')
        #  Semantic rubric matching
        rubric_res = check_rubric_concepts(stu_clean, rubric_rules, use_semantic=True)
        
        # Stop here if just extracting features for training
        if extract_features_only:
            return {
                "components": {
                    "math_sym_score": math_res.get('math_score', 0.0) or 0.0,
                    "nli_entailment": semantic_score,
                    "rubric_coverage": rubric_res.get('rubric_coverage', 0.0) or 0.0
                }
            }
        
        print(f'6. Fusing scores...')
        
        # Use trained weights if available
        custom_weights = None
        if self.use_trained_weights and self.trainer.optimal_weights:
            custom_weights = self.trainer.optimal_weights
            print(f'  Using trained weights: {custom_weights}')
        
        #  Transparent weight reporting
        report = grade_fused(math_res, semantic_score, rubric_res, q_type, 
                           pass_thresh, custom_weights)
        report["time_sec"] = round(time.time() - t0, 2)
        report["question_type_detected"] = q_type
        
        # Add debug payload
        report["debug_info"] = {
            "1_raw_ocr": raw_ocr,
            "2_questions_detected": len(questions),
            "3_cleaned_student_text": stu_clean,
            "4_math_engine_matches": math_res.get("details", []),
            "5_nli_entailment_prob": semantic_score,
            "6_rubric_matches": rubric_res.get("concept_scores", {})
        }
        
        clear_output()
        return report

print('✅ Fixed Grading Pipeline ready.')

import json

# Instantiate pipeline
pipe = GradingPipeline()

# Example grading


IMAGE_PATH = '/teamspace/studios/this_studio/.lightning_studio/studentHandwritten-vYDGOUFs (1).jpeg'

REFERENCE_TEXT = """
1. Dukkha : Human life involves suffering, pain..
2. Samudaya: Suffering arises due to desire, craving..
"""
RUBRIC = [
    # {"concept": "INgestion", "requires_negation": False},
    # {"concept": "OCR", "requires_negation": False},
    # {"concept": "Aryan Soni", "requires_negation": False},
]
# Grade (auto-detects question type)
report = pipe.grade(IMAGE_PATH, REFERENCE_TEXT, RUBRIC)

# Display results
print('=' * 60)
print('           FINAL GRADE REPORT')
print('=' * 60)

report_copy = report.copy()
debug_payload = report_copy.pop("debug_info", {})

print(json.dumps(report_copy, indent=2))

print('\n' + '=' * 60)
print('🛠️        ENGINE DEBUG INSPECTION')
print('=' * 60)
print(json.dumps(debug_payload, indent=2))

# Prepare training data
training_examples = [
    {
        'image_path': '/path/to/student1.png',
        'reference': REFERENCE_TEXT,
        'rubric': RUBRIC,
        'human_score': 8.5,  # Human grader gave 8.5/10
        'question_type': 'theory'
    },
    {
        'image_path': '/path/to/student2.png',
        'reference': REFERENCE_TEXT,
        'rubric': RUBRIC,
        'human_score': 6.0,
        'question_type': 'theory'
    },
    # Add more examples...
]

# Extract features
X, y = pipe.trainer.collect_training_data(pipe, training_examples)

# Train optimal weights
optimal_weights = pipe.trainer.train_weights(X, y, question_type='theory')

# Train pass/fail classifier
classifier = pipe.trainer.train_classifier(X, y, threshold=5.0)

# Save trained models
pipe.trainer.save('/home/claude/trained_grader.pkl')

# Use trained weights for grading
pipe.use_trained_weights = True
report_trained = pipe.grade(IMAGE_PATH, REFERENCE_TEXT, RUBRIC)

print("\n📊 Grading with trained weights:")
print(json.dumps(report_trained, indent=2))
