# Paper Draft — Hybrid SVM-LLM for Ticket Classification

> **Status:** Draft v1 (~2026-05-11)
> **Target venue:** ICITRI 2026 (conference) + Q2 journal extended version
> **Word count target:** ~5500-6500 (journal)

---

## Title Options (pick 1)

1. *"Hybrid SVM-LLM for IT Helpdesk Ticket Classification: Feature Fusion Wins, Decision Voting Marginal, and Label Space Size is the Hidden Lever"*
2. *"When Constrained Shortlist Beats Free-Form Reasoning: An Empirical Study on Hybrid SVM-LLM Architectures for Imbalanced Ticket Classification"*
3. *"Three Faces of Hybrid SVM-GenAI: A Comparative Study of Feature Fusion, Decision Correction, and Voting Ensemble"*

**Recommended:** #1 (concrete, mentions 3 contributions)

---

## Abstract

*~250 words, structured: Background → Methods → Results → Conclusion*

Customer support ticket classification is essential for IT helpdesk automation but faces challenges from severe class imbalance and ambiguous natural language descriptions. Recent advances in Generative AI (GenAI) offer two integration strategies with classical machine learning: **feature-level fusion** (combining ML features with neural embeddings) and **decision-level integration** (using LLM to correct or vote on ML predictions). However, the relative effectiveness of these strategies, and the role of prompt design in decision-level hybrids, remain underexplored.

We present a comprehensive empirical study on a customer support ticket dataset (16,338 samples, 19 categories, 3 priority levels) comparing four baselines (SVM, Random Forest, Logistic Regression, BERT) against three hybrid SVM-GenAI architectures: (i) **Feature Fusion** (TF-IDF + OpenAI Embedding → LinearSVC), (ii) **Decision Voting** (3-way majority: SVM + Fusion + LLM voter), and (iii) **Decision Correction** (LLM refines SVM predictions). We evaluate via stratified 80/20 train/test split with 5-fold cross-validation.

**Results:** Feature Fusion significantly outperforms SVM (paired t-test, p<0.01) with +0.89% accuracy and +2.03% macro F1 on category classification. Decision Voting yields only marginal gain (+0.15%) at 100× higher cost. Critically, through systematic prompt ablation (5 variants × 300 samples) and label space sweep (K=3 to 19), we find that **anchor framing in prompts does not significantly affect LLM correction quality**; instead, **constraining the candidate label space to top-K ML predictions is the key lever**, with a monotonic 11-percentage-point drop in LLM correctness as K increases from 3 to 19.

**Conclusion:** Practitioners should prefer feature-level fusion for cost-effective hybrid ML-LLM systems. When decision-level LLM integration is used, constrain output space to top-3 ML candidates for optimal correction quality.

**Keywords:** text classification, hybrid ML-LLM, feature fusion, prompt engineering, label space, IT helpdesk, customer support

---

## 1. Introduction

### 1.1 Motivation

IT helpdesk teams handle thousands of customer support tickets daily, each requiring routing to the correct team (category) and prioritization based on urgency. Manual classification is labor-intensive and error-prone, especially as ticket volume scales. Automated text classification can reduce response time and operational cost, but faces three challenges:

1. **Class imbalance** — popular categories (e.g., Security, Bug) dominate while specialized categories (Hardware, Customer Support) are rare.
2. **Ambiguous natural language** — users describe issues with synonyms, parafrase, multi-issue descriptions.
3. **Domain shift** — generic LLMs may not capture organization-specific terminology.

Classical ML (SVM, Random Forest) with TF-IDF features is fast and interpretable but limited to lexical patterns. Deep learning baselines (BERT, DistilBERT) capture semantic information but require expensive fine-tuning, which is impractical for many organizations without GPU resources.

Recent work in **hybrid ML-LLM systems** offers a middle ground: combine classical ML with pre-trained GenAI components (embeddings, LLMs) accessible via API. Two integration strategies are common:

- **Feature-level fusion:** concatenate ML features (TF-IDF) with neural embeddings (e.g., OpenAI's `text-embedding-3-small`) and feed into a classical classifier.
- **Decision-level integration:** use an LLM (e.g., GPT-4) to correct or vote on ML predictions via prompt engineering.

### 1.2 Research Questions

This paper investigates three research questions:

- **RQ1:** Does Hybrid SVM-GenAI (any architecture) outperform single SVM on imbalanced ticket classification, with statistical significance?
- **RQ2:** Among hybrid architectures (Fusion, Voting, Correction), which provides the best cost-performance trade-off?
- **RQ3:** What design choices in decision-level LLM integration (prompt anchor framing vs. candidate label space size) most affect correction quality?

### 1.3 Contributions

1. **(Main)** Empirical evidence that Hybrid Feature Fusion (TF-IDF + OpenAI Embedding → SVM) significantly outperforms single SVM (paired t-test p<0.01) on a 16,338-sample dataset.
2. **(Cost analysis)** Demonstration that Hybrid Decision Voting yields only marginal improvement (+0.15% Acc) over Fusion at 100× higher cost, making Fusion more cost-effective.
3. **(Novel insight)** Through systematic ablation, we reframe the conventional wisdom that "anchor framing matters" in hybrid prompts: **label space size is the actual lever**. Monotonic correction quality decrease from K=3 to K=19 candidate labels (11 percentage points). Practical recommendation: constrain LLM to top-3 ML predictions.

### 1.4 Paper Structure

Section 2 reviews related work on text classification, hybrid ML-LLM systems, and prompt engineering. Section 3 describes our dataset and methodology. Section 4 presents experimental setup and results. Section 5 discusses implications, limitations, and future work. Section 6 concludes.

---

## 2. Related Work

### 2.1 Text Classification Methods

Classical methods like SVM with TF-IDF [REF: Joachims 1998] remain strong baselines for text classification despite the rise of deep learning. Random Forest and Logistic Regression are also commonly used. These methods are fast, interpretable, and require no GPU.

Deep learning baselines like BERT [REF: Devlin 2019], DistilBERT [REF: Sanh 2019], and RoBERTa [REF: Liu 2019] achieve state-of-the-art on many text classification benchmarks but require fine-tuning on labeled data, which is GPU-intensive.

### 2.2 Hybrid ML-LLM Systems

Recent work explores combining classical ML with LLMs. **Feature-level fusion** approaches concatenate TF-IDF or count-based features with neural embeddings before feeding into a classifier [REF: ...]. This leverages the semantic understanding of LLMs without losing the precision of lexical features.

**Decision-level integration** uses LLMs as post-processing components. Strategies include:

- LLM-as-validator (LLM checks ML prediction)
- LLM-as-corrector (LLM may override ML prediction)
- LLM-as-voter (majority vote among multiple predictors)

[REF: prior work on LLM voting, e.g., self-consistency, ensemble methods]

### 2.3 Prompt Engineering and Anchor Bias

LLM behavior is sensitive to prompt design. Studies have observed:

- **Anchor bias / sycophancy** [REF: Sharma et al. 2023]: LLM tends to agree with information given in prompt
- **Few-shot examples** improve performance [REF: Brown 2020]
- **Chain-of-thought** prompts improve reasoning [REF: Wei 2022]

In hybrid ML-LLM contexts, some authors hypothesize that mentioning ML prediction in the prompt anchors LLM to that prediction, reducing correction effectiveness. We test this hypothesis empirically and find the conventional wisdom does not hold.

---

## 3. Methodology

### 3.1 Dataset

We use the **COBACEK customer support dataset** (synthetic, English, from Kaggle [REF: dataset URL]), consisting of 16,338 IT helpdesk tickets. Each ticket has:

- `description`: free-text problem description (avg 368 characters)
- `priority`: 3 levels (low, medium, high)
- `category_filtered`: 19 merged categories (from 81 original fine-grained labels)

Category distribution is **imbalanced**: Security (3,333 samples) vs. Customer Support (67 samples), a 50× ratio.

**Train/test split:** Stratified 80/20 by category, `random_state=42`. Train: 13,070, Test: 3,268.

### 3.2 Base Models

We compare four baselines:

| Model | Features | Classifier |
|---|---|---|
| SVM | TF-IDF (unigram + bigram) | LinearSVC |
| Random Forest | TF-IDF | RandomForest (n=200) |
| Logistic Regression | TF-IDF | LR (max_iter=1000) |
| BERT | DistilBERT-multilingual | Fine-tuned classifier head |

### 3.3 Hybrid SVM-GenAI Architectures

#### 3.3.1 Hybrid Feature Fusion

```
description → [TF-IDF (50k dim)] ─┐
            → [OpenAI Embedding (1536 dim)] ─┐
                                              ├─→ concat → LinearSVC → label
```

OpenAI Embedding from `text-embedding-3-small` (1536-dim semantic vector). Features concatenated before classifier.

#### 3.3.2 Hybrid Decision Voting (3-way)

For each test sample, three voters predict:

1. SVM (base)
2. Hybrid Fusion (above)
3. LLM Voter — `gpt-4.1-mini` predicts independently via prompt (no ML hint)

Final prediction = majority vote. Tie-break to Fusion (strongest single voter).

#### 3.3.3 Hybrid Decision Correction

For each test sample where SVM confidence is low:

1. SVM predicts → top-K candidates extracted
2. LLM (gpt-4.1-mini) is prompted with ticket text + top-K candidates
3. LLM selects one label from top-K (constrained)

We study this architecture extensively in Section 4.4 (ablation).

### 3.4 Evaluation Protocol

- **Single split:** stratified 80/20, `random_state=42`
- **5-fold CV:** Stratified, base_seed=42, fold_i uses seed=42+i
- **Metrics:** Accuracy, macro precision/recall/F1, weighted F1
- **Statistical test:** paired t-test across 5 folds (Hybrid Fusion vs SVM)
- **Focus metric:** macro F1 (fairness to minority classes)

---

## 4. Experiments

### 4.1 Main Result: Hybrid Fusion vs Baselines (5-Fold CV)

**Table 1: 5-Fold Stratified CV Results (mean ± std)**

| Model | Acc Cat | F1 Cat (macro) | Acc Pri | F1 Pri (macro) |
|---|---|---|---|---|
| **Hybrid Fusion** ✅ | **0.8214 ± 0.003** | **0.6738 ± 0.009** | **0.7219 ± 0.009** | **0.7097 ± 0.008** |
| SVM | 0.8125 ± 0.004 | 0.6535 ± 0.015 | 0.7167 ± 0.011 | 0.7033 ± 0.009 |
| Random Forest | 0.7660 ± 0.005 | 0.5353 ± 0.018 | 0.7172 ± 0.009 | 0.6838 ± 0.010 |
| Logistic Regression | 0.7764 ± 0.005 | 0.4713 ± 0.007 | 0.6335 ± 0.008 | 0.5864 ± 0.008 |

**Paired t-test (per-fold):**

- Acc Cat: Hybrid Fusion > SVM, **p = 0.007** (signifikan p<0.01)
- F1 Cat: Hybrid Fusion > SVM, **p = 0.004** (signifikan p<0.01)
- Konsisten unggul di **semua 5 fold**

**Answer to RQ1:** Yes, Hybrid Fusion outperforms SVM with statistical significance.

### 4.2 Hybrid Voting Analysis (3-way Ensemble)

We run Hybrid Voting with gpt-4.1-mini on the test set (3,268 samples × 1 GenAI call = ~3 hours, ~$3).

**Table 2: Three Architectures Comparison (single split)**

| Approach | Acc Cat | F1 Cat | Cost |
|---|---|---|---|
| SVM | 0.8146 | 0.6698 | Free |
| Hybrid Fusion | 0.8250 | 0.6881 | $0.03 + 5 min |
| GenAI Voter alone | 0.5346 | 0.4274 | $3 + 3 h |
| **Hybrid Voting (3-way)** | **0.8265** | **0.6888** | $3 + 3 h |

**Key observations:**

- GenAI Voter alone (Acc 0.5346) is much weaker than SVM (0.8146) — LLM lacks domain-specific training.
- Hybrid Voting marginally outperforms Fusion (+0.15% Acc, +0.07% F1) — within noise.
- Voting agreement: in 99.2% of cases, the voting outcome equals Fusion's prediction. **The Voter's prediction is almost always outvoted by SVM and Fusion**, which usually agree.
- **Cost-efficiency:** Fusion provides ~85% of total improvement at ~1% of the cost.

**Answer to RQ2:** Hybrid Fusion provides the best cost-performance trade-off. Voting is not worth the additional cost.

### 4.3 Anchor Bias Ablation (5 prompt variants)

We hypothesize whether LLM correction is sensitive to **anchor framing** in prompts. We design 5 prompt variants and test each on the same 300 SVM-wrong samples (1,500 LLM calls total, $2):

| Variant | Description |
|---|---|
| V1 NO_ML | No mention of ML prediction (control) |
| V2 NEUTRAL_ML | ML prediction mentioned neutrally |
| V3 DEFER_ML | "Current ML prediction: X. Improve only if needed." (strong anchor) |
| V4 CHALLENGE_ML | "ML may be wrong; verify independently." (anti-anchor) |
| V5 TOP3_CHOICES | Multiple-choice from top-3 SVM candidates (constrained) |

**Table 3: Anchor Bias Ablation Results (N=300 per variant)**

| Variant | Override Rate | Correction Rate | LLM Correct Rate |
|---|---|---|---|
| V1 NO_ML | 52.3% | 8.0% | 45.3% |
| V2 NEUTRAL_ML | 48.7% | 7.0% | 44.7% |
| V3 DEFER_ML | 54.0% | 8.0% | 43.3% |
| V4 CHALLENGE_ML | 51.7% | 6.7% | 43.7% |
| **V5 TOP3_CHOICES** ⭐ | **37.0%** | **12.7%** | **56.7%** |

**Key finding:** V1-V4 produce nearly identical outcomes (LLM correct rate 43-45%). **Anchor framing has minimal effect.** V5 (constrained shortlist) is dramatically better.

This contradicts the common belief that anchor framing causes LLM-as-corrector to fail.

### 4.4 Label Space Size Ablation (Validation)

To validate the hypothesis that **label space size**, not anchor framing, is the lever, we run V5-style prompt with varying K (3, 5, 7, 10, 19) on the same samples (1,500 calls).

**Table 4: Top-K Label Space Ablation (N=300)**

| K | Override Rate | Correction Rate | LLM Correct Rate |
|---|---|---|---|
| **3** | 35.3% | **12.0%** | **56.3%** |
| 5 | 39.0% | 9.3% | 52.7% |
| 7 | 41.3% | 9.3% | 52.3% |
| 10 | 45.7% | 9.3% | 49.7% |
| 19 (all) | 51.7% | 7.0% | 45.3% |

**Monotonic decrease:** as K increases from 3 to 19 (full label space), LLM Correct Rate drops from 56.3% to 45.3% (11 percentage points, ~24% relative drop).

**Answer to RQ3:** Label space size is the dominant factor. Constraining the candidate set significantly improves correction quality.

[Figure 1: Line chart showing monotonic relationship K vs. LLM correct rate. Available at `results/figures_phase3/topk_ablation.png`]

---

## 5. Discussion

### 5.1 Why Feature Fusion Works

TF-IDF captures lexical patterns (specific terms like "VPN", "outage"), while OpenAI Embedding captures semantic patterns (synonyms, paraphrases, intent). They are **complementary**: when one is uncertain, the other helps disambiguate. SVM combining both has more discriminative information than either alone.

### 5.2 Why Voting Adds Little

Voting works when voters have **uncorrelated errors**. In our setup:

- SVM and Fusion errors are highly correlated (Fusion is essentially "enhanced SVM")
- LLM Voter is weak in domain (Acc 0.53), so its votes are often outvoted

For voting to add value, voters must be **independent and reasonably strong**. A weak third voter cannot rescue an already-agreeing pair.

### 5.3 Why Constrained Shortlist Matters

Counter to intuition, anchor framing in prompts does not strongly affect LLM correction. The real challenge for LLM is **the candidate space**: with 19 possible categories, the LLM must reason through many options, increasing error rate. With top-3 constraint, the LLM only chooses among the most plausible candidates (typically 80%+ correct already in top-3), making the task easier.

This is analogous to **multiple-choice vs free-response** in human exams: constrained format reduces error rate.

### 5.4 Practical Recommendations

For practitioners building hybrid ML-LLM ticket classification:

1. **Use feature-level fusion** as the primary integration strategy. It's cheap (~$0.03/run for 16K samples), fast (~5 minutes), and effective.
2. **Skip decision-level voting** unless you have access to a strong, domain-tuned LLM. With generic LLMs, voting adds cost without value.
3. **If using LLM correction**, always constrain output to top-K ML candidates (recommended K=3). Do not pass the full label space.

### 5.5 Limitations

- **Single dataset:** Findings may not generalize to other domains. Future work should validate on multilingual or specialized helpdesk datasets.
- **OpenAI dependency:** Reliance on OpenAI API limits reproducibility for organizations without API access. Open-source alternatives (e.g., Sentence-BERT) should be evaluated.
- **Single LLM family:** All experiments use OpenAI models. Validation with Anthropic Claude, Google Gemini, or open-source LLMs (Llama) is needed.
- **Synthetic dataset:** COBACEK is synthetic; real production tickets may have different characteristics (noise, code snippets, multilingual mix).

---

## 6. Conclusion

We presented a comprehensive empirical study on hybrid SVM-LLM architectures for IT helpdesk ticket classification on a 16,338-sample dataset. We make three contributions:

1. Feature-level Hybrid Fusion significantly outperforms SVM (paired t-test p<0.01) at minimal cost.
2. Decision-level Hybrid Voting provides only marginal gains at 100× cost, making it cost-inefficient.
3. The conventional wisdom that "anchor framing causes LLM correction failure" is incorrect; the actual lever is **label space size** (monotonic relationship K=3 → K=19, -11pp LLM accuracy).

For practitioners: prefer feature-level fusion, constrain LLM correction to top-3 ML candidates.

**Future work** includes (i) validation across multiple datasets and LLM families, (ii) cost-aware adaptive hybrid systems that dynamically choose architecture per sample, (iii) integration with retrieval-augmented generation (RAG) for domain adaptation.

---

## References

> *Placeholder — needs full citation list. Will use BibTeX in final manuscript.*

- [1] Joachims, T. (1998). *Text categorization with support vector machines*. ECML.
- [2] Devlin, J. et al. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding*. NAACL.
- [3] Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT*. arXiv:1910.01108.
- [4] Liu, Y. et al. (2019). *RoBERTa: A robustly optimized BERT pretraining approach*. arXiv:1907.11692.
- [5] Brown, T. et al. (2020). *Language models are few-shot learners*. NeurIPS.
- [6] Wei, J. et al. (2022). *Chain-of-thought prompting elicits reasoning in large language models*. NeurIPS.
- [7] Sharma, M. et al. (2023). *Towards understanding sycophancy in language models*. arXiv:2310.13548.
- [8] [OpenAI text-embedding-3-small documentation]

---

## Appendix

### A. Reproducibility

All code, data, and results are available at: `https://github.com/mahathirmuh/it-helpdesk-ticket-classifier` (branch: `hybrid-improvement`).

```bash
# Main pipeline (5-fold CV)
python src/compare_svm_genai.py --skip-bert --n-folds 5

# Hybrid Voting Ensemble
python src/compare_svm_genai.py --skip-bert --enable-voting --model gpt-4.1-mini

# Anchor bias ablation (Tahap 3)
python src/anchor_bias_ablation.py --n-samples 300

# Top-K label space sweep (Tahap 3b)
python src/topk_ablation.py --n-samples 300 --topk-list 3,5,7,10,19
```

### B. Computational Cost

| Phase | OpenAI Cost | Wall Time |
|---|---|---|
| Embedding (per run) | ~$0.03 | ~5 min |
| Voting (per LLM, per run) | ~$3 | ~3 h |
| Anchor bias ablation (1500 calls) | ~$2 | ~1.5 h |
| Top-K ablation (1500 calls) | ~$2 | ~45 min |
| **Total study** | **~$10** | **~10 h** |

### C. Per-Class F1 Analysis

[Table/Figure showing per-class F1 for SVM vs Fusion, especially for minority classes]
[Available at `results/analysis_phase1.xlsx` sheet `cat_per_class_f1`]
