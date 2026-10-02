# Project 1 — Credibility Scoring for Sources

**CS676 Algorithms for Data Science • Pace University, Seidenberg School**

## 📌 Quick Links

| Link | Purpose |
|------|---------|
| **[🚀 Live App](https://project1-kry9wijpxj8nub4otg2psz.streamlit.app/)** | Try the credibility scorer live |
| **[📄 Technical Report](./cs676_report.pdf)** | Full algorithm & results (4 pages) |
| **[🎬 Video Demo](./livedemo.mp4)** | 2-minute walkthrough |
| **[📊 Professor's Repo](https://github.com/yiqiao-yin/pace-u-cs676)** | Original project template |

---

## 🎯 The Problem

How do we programmatically determine if a source is trustworthy? This project builds a **hybrid credibility scoring system** that combines rule-based analysis with LLM evaluation to rate URLs on a scale of 0–1.

**Weight:** 30% of course grade | **Points:** 100 + 5% bonus for public deployment 

---

## 🔧 My Implementation —  4 Improvements

### 1. **Expanded Domain List** 
Grew from ~30 domains to **55+ domains** with hand-tuned credibility scores:
- **Medical journals:** JAMA (0.92), BMJ (0.93), PLOS (0.85), eLIFE (0.88)
- **Medical preprints:** medRxiv (0.60), PsyArXiv (0.60)
- **News outlets:** BBC (0.82), Guardian (0.80), CNN (0.75), Fox News (0.70)
- **Government:** CDC (0.92), NIH (0.92), WHO (0.90), FDA (0.90)

### 2. **Crossref API Integration**
Extracts DOIs and queries the free Crossref API for:
- **Retracted papers:** -0.50 penalty
- **Citation count:** 500+ citations → +0.15, 100–500 → +0.08

### 3. **Learned Weights** (Logistic Regression)
- Trained on 24 labeled URLs → **91.7% accuracy**
- Optimal weights: Rule-based 67% + LLM 33% (not guessed 60/40)

### 4. **Subdomain Penalty**
Detects personal URLs on trusted domains:
- Patterns: `blog`, `~user`, `/home/`, `my-`, etc.
- Penalty: -0.15 for personal subdomains on otherwise trusted sites

---

## 📈 Results

| Metric | Baseline | After All 4 | With LLM |
|--------|----------|-------------|----------|
| **MAE** | 0.142 | 0.108 | 0.076 |
| **Band Accuracy** | 66.7% | 75.0% | 87.5% |
| **Worst Error** | 0.410 | 0.320 | 0.230 |
| **Improvement** | — | **24% better** | **46% better** |

**Key fix:** JAMA article improved from 0.52 → 0.93

---

## 📁 What's In This Repo

- **`credibility.py`** — Main scoring function (27 KB, all 4 improvements)
- **`main.py`** — Streamlit web app for live testing
- **`evaluate.py`** — Evaluation metrics (MAE, band accuracy, etc.)
- **`learn_weights.py`** — Logistic regression weight optimization
- **`test_credibility.py`** — 26 unit tests (all passing ✅)
- **`cs676_report.pdf`** — Technical report with literature review
- **`livedemo.mp4`** — 2-minute video walkthrough
- **`requirements.txt`** — All dependencies (requests, scikit-learn, Streamlit, Anthropic)

---

## 🚀 Try It Live

No setup needed! Visit: **[Project 1 Credibility Scorer](https://project1-kry9wijpxj8nub4otg2psz.streamlit.app/)**

Paste any URL and get instant credibility feedback with reasoning.

---

## 🛠 Run Locally

### Setup
```bash
# Clone the repo
git clone https://github.com/kel-den/cs676-kelden.git
cd cs676-kelden/projects/project_1

# Install dependencies
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY=sk-ant-...
```

### Run the App
```bash
streamlit run main.py
```
Then open `http://localhost:8501`

### Run Tests
```bash
python -m pytest test_credibility.py -v
```
Expected: **26 passed** ✅

### Evaluate Performance
```bash
python evaluate.py
```
Shows MAE, band accuracy, and detailed error analysis.

---

## 📚 Key Test Cases (All Passing)

```
✅ Preprints score lower than peer-reviewed journals
✅ JAMA (medical journal) scores ≥ 0.85
✅ WHO.int (trusted gov) scores ≥ 0.80
✅ Personal blogs score lower than BBC
✅ HTTPS scores higher than HTTP
✅ Retracted papers penalized
```

---

## 🎓 Technical Approach

The system uses a **hybrid pipeline**:

1. **Rule-based scoring** (67% weight)
   - Domain reputation lookup
   - URL structure analysis (HTTPS, subdomain check)
   - Metadata extraction (DOI, citations)

2. **LLM evaluation** (33% weight)
   - Claude analyzes page content
   - Considers writing quality, citations, bias

3. **Combine with learned weights** → Final score (0–1)

See `cs676_report.pdf` for full algorithm description, literature review, and failure analysis.

---

## 📝 Notes

- **Anthropic API Key Required:** Set `ANTHROPIC_API_KEY` environment variable or in `.env` file
- **Streamlit Cloud:** App deployed with Python 3.11 (pre-built pyarrow wheels)
- **Rate Limits:** Crossref API is free and open; no authentication needed
- **Bonus:** +5% for public Streamlit deployment ✅

---

## 🤝 Credit

Project template and starter code by [Prof. Yiqiao Yin](https://github.com/yiqiao-yin).  
Improvements, deployment, and evaluation by Kelden T.

---

## 📬 Questions?

Check the technical report (`cs676_report.pdf`) or watch the video demo (`livedemo.mp4`) for a walkthrough.
