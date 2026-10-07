\# End-to-End Telecom Churn \& Financial ROI Production Pipeline



A production-grade, modular machine learning lifecycle pipeline designed to extract consumer data over the wire from a cloud-hosted relational layer, execute transformations dynamically, and evaluate business retention profit using an optimized XGBoost ensemble engine.



\## 🛠️ Technical Architecture \& Ecosystem

\* \*\*Infrastructure Layer\*\*: Cloud-Hosted Relational Database (Supabase PostgreSQL Instance).

\* \*\*Data Transport Protocols\*\*: Native Wire Protocol connection via isolated configuration parameters (`urllib.parse` + `pg8000`).

\* \*\*Processing Core\*\*: Modular Scikit-Learn `ColumnTransformer` \& `Pipeline` configurations to prevent data leaks.

\* \*\*Predictive Framework\*\*: `XGBoost Classifier` tuned defensively with scale-invariant imbalance handling.

\* \*\*Artifact Serialization\*\*: Static binary arrays packed via `joblib` for high-throughput cloud inferencing.

\* \*\*Frontend Web Application\*\*: Streamlit Engine wrapped inside a custom container configuration.



\## 📂 Repository Directory Layout

\* `src/data/` - SQL fetching scripts and connection handlers.

\* `src/features/` - Automated preprocessing pipelines and data cleaning scripts.

\* `src/models/` - Training orchestration code and serialized binary models (`.joblib`).

\* `src/app/` - Web presentation layers and financial ROI calculation engines.

\* `Dockerfile` - Container virtualization configuration layout.



\## 🎯 Validation Metrics \& Business ROI Impact

\* \*\*ROC-AUC Evaluation Score\*\*: `84.34%`

\* \*\*True Positive Capture (Recall)\*\*: `81.00%` (Imbalance Resilient)

\* \*\*Financial Protection Capacity\*\*: Effectively captures over 81% of canceling accounts, translating raw model probabilities into actionable gross margins (e.g., protecting over ₹11 Lakhs in gross revenue depending on campaign mechanics).



