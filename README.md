# GrieveAI

## Explainable Near-Real-Time Hindi-English Code-Mixed Grievance Triage and Department Routing

GrieveAI is an AI-assisted grievance triage and routing system designed for higher-education institutions.

The project focuses on processing student grievances written in:

- English
- Devanagari Hindi
- Romanised Hindi / Hinglish
- Hindi-English code-mixed text

The system is being developed as a case study for the **VCET Student Grievance Redressal workflow**.

---

## Project Objective

The goal of GrieveAI is not simply to classify a grievance.

The system is designed to transform raw student complaints into structured, explainable and actionable information that can support the institutional grievance redressal workflow.

A typical workflow is:

Student grievance
        ↓
Language / script detection
        ↓
PII detection and redaction
        ↓
Category classification
        ↓
Subcategory classification
        ↓
Priority estimation
        ↓
Confidence gate
        ↓
Duplicate / recurring grievance detection
        ↓
Explanation
        ↓
Summary and action recommendation
        ↓
Department routing
        ↓
Ticket / SLA tracking
        ↓
Administrator dashboard
        ↓
Resolution and audit trail

Human administrators remain responsible for final decisions.

---

# Current Taxonomy

GrieveAI currently uses **7 major categories and 33 subcategories**.

## 1. Academics

- `timetable_scheduling`
- `teaching_quality`
- `faculty_conduct`
- `syllabus_course_progress`
- `attendance`

## 2. Examinations

- `hall_ticket`
- `marks_discrepancy`
- `revaluation`
- `examination_schedule`
- `examination_process`

## 3. Fees / Accounts

- `fee_discrepancy`
- `refund`
- `scholarship`
- `payment_transaction`
- `accounts_financial_documentation`

## 4. IT / Library

- `portal_account_access`
- `wifi_network`
- `software_license`
- `library_book_availability`
- `library_digital_resources`

## 5. Infrastructure

- `classroom_lab_maintenance`
- `electrical_issues`
- `sanitation_cleanliness`
- `parking_access`
- `construction_facility_disruption`

## 6. Transport

- `commuting_accessibility`
- `transport_academic_conflict`
- `travel_safety_access`

## 7. Canteen

- `food_quality`
- `food_hygiene`
- `pricing`
- `menu_variety`
- `food_service`

Cross-domain grievances are intended to be represented through a primary issue together with contextual information such as contributing causes, affected areas and recommended actions.

---

# Machine Learning

The current primary transformer model is:

**IndicBERT v2 + LoRA**

Model:

`ai4bharat/IndicBERTv2-MLM-only`

The model is fine-tuned using parameter-efficient LoRA adapters.

The architecture contains:

- Category classification head
- Hierarchical subcategory classification head
- Priority regression head

Subcategory prediction is constrained according to the selected category.

---

# Model Experiments

The project has evaluated multiple approaches.

| Approach | Category Macro-F1 | Subcategory Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 86.4% | 69.5% |
| MuRIL + LoRA | 98.6% | 30.5% |
| MuRIL category + TF-IDF subcategory | 98.6% | 69.6% |
| IndicBERT v2 + LoRA | 100.0% | 91.2% |

The IndicBERT result above comes from the synthetic development/test dataset and should not be interpreted as VCET pilot performance.

---

# 60K Stress Test

A larger synthetic dataset containing:

- 60,000 records
- 7 categories
- 33 subcategories
- English
- Devanagari Hindi
- Romanised Hinglish

was used to test training scalability.

The 3-epoch IndicBERT v2 + LoRA stress test produced:

- Category Macro-F1: **100.0%**
- Subcategory Macro-F1: **100.0%**
- Priority MAE: **0.783**
- High-priority recall: **0.0%**

These results demonstrate scalability on the synthetic benchmark.

They are **not representative of real-world VCET performance** because the dataset is artificially generated.

---

# Why IndicBERT v2?

The project requires multilingual and code-mixed processing involving Indian languages and English.

IndicBERT v2 is therefore being evaluated as the primary language representation model because its pretraining is targeted toward Indian-language NLP.

The final deployment model will be selected based on evaluation using human-annotated VCET data rather than synthetic data alone.

---

# Baseline and Research Comparisons

The project maintains multiple model approaches for comparison:

### Classical baseline

TF-IDF features with Logistic Regression provide a strong and interpretable baseline.

### MuRIL

MuRIL + LoRA was evaluated for multilingual Indian-language representation.

### IndicBERT v2

IndicBERT v2 + LoRA currently provides the strongest synthetic subcategory results among the tested transformer approaches.

Keeping these experiments allows the final study to compare classical and transformer-based approaches rather than reporting only a single model.

---

# Planned Explainability

GrieveAI is intended to provide explanations for classification decisions.

The planned approach uses SHAP-based token attribution so that administrators can inspect which parts of a grievance contributed to a prediction.

The goal is to preserve the original language/script of the grievance where practical.

---

# Duplicate and Recurring Grievance Detection

The system is intended to identify semantically similar grievances.

This allows multiple individual complaints to be grouped into recurring institutional issues.

For example:

    42 students report problems with the same Wi-Fi service.

Instead of treating these only as 42 independent tickets, the system can identify the recurring issue and provide administrators with an aggregated view.

---

# LLM-Assisted Analysis

The LLM layer is separate from the primary classifier.

IndicBERT performs structured classification locally.

An LLM is intended for tasks such as:

- Grievance-group summarisation
- Root-cause/theme extraction
- Administrative report generation
- Action recommendations
- Natural-language summaries

The intended architecture uses an OpenRouter-compatible API with an open-weight LLM.

PII redaction will occur before sensitive grievance content is sent to an external LLM service.

LLM recommendations are advisory and remain subject to human review.

---

# Backend and Application

Planned backend:

- Python
- Flask
- Flask-SQLAlchemy
- PostgreSQL

Planned application components:

- Student grievance submission
- Grievance processing pipeline
- Admin dashboard
- Classification results
- Confidence indicators
- SHAP explanations
- Duplicate/recurring grievance groups
- Department routing
- Ticket management
- SLA tracking
- Human override
- Audit trail
- Report generation

---

# Database

PostgreSQL is planned as the primary production database.

The database will store structured grievance information such as:

- Grievance text / appropriately protected representation
- Language
- Category
- Subcategory
- Priority
- Confidence
- Duplicate / recurring group
- Routing recommendation
- Ticket status
- SLA information
- Human overrides
- Audit information
- Model version

The trained ML model itself is stored separately from PostgreSQL.

---

# Model Deployment Concept

The trained IndicBERT model will be stored locally with the application.

Conceptually:

    Student
       ↓
    Flask application
       ↓
    Local IndicBERT model
       ↓
    Classification
       ↓
    PostgreSQL
       ↓
    Admin dashboard

The model will not automatically retrain whenever a new grievance arrives.

Instead, corrected and human-validated grievances can later be collected into a new training dataset and used for controlled model updates.

Model versions will be maintained separately.

---

# Real VCET Data

Synthetic data is currently being used for development and experimentation.

The next major research stage is evaluation using de-identified and appropriately authorized VCET grievance data.

The intended process is:

1. Obtain appropriate institutional permission.
2. De-identify sensitive information.
3. Establish the final annotation guidelines.
4. Human-annotate a representative dataset.
5. Create train/validation/test splits.
6. Fine-tune IndicBERT v2.
7. Evaluate on held-out real VCET grievances.
8. Perform error analysis.
9. Evaluate confidence-based human handoff.
10. Evaluate department routing.
11. Conduct pilot testing.
12. Document results for the research paper.

Synthetic results will not be presented as real VCET performance.

---

# Evaluation Plan

The final evaluation will include more than classification accuracy.

Planned metrics include:

### Classification

- Category Macro-F1
- Subcategory Macro-F1
- Per-class precision
- Per-class recall
- Confusion matrices

### Priority

- Priority MAE
- High-priority recall

### Workflow

- Department routing accuracy
- Human override rate
- Confidence-gate performance
- Duplicate/recurring grievance detection
- Time-to-route reduction

### LLM output

- Summary factuality
- Human evaluation
- Action recommendation usefulness

---

# Research Positioning

The project is positioned around the combination of:

- Higher-education grievance processing
- Hindi-English code-mixed text
- Fine-grained hierarchical classification
- Explainability
- Confidence-aware human handoff
- Duplicate/recurring grievance aggregation
- LLM-assisted administrative analysis
- Institutional department routing
- End-to-end grievance workflow evaluation

The research contribution will ultimately be evaluated using real, appropriately authorized and human-annotated institutional data.

---

# Repository Structure

```text
GrieveAI/
│
├── config/
│   └── taxonomy.json
│
├── data/
│   └── README.md
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── baseline_classifier.py
│   └── ...
│
├── tests/
│
├── experiments/
│   ├── README.md
│   └── notebooks/
│
├── checkpoints/
│   └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore