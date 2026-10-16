"""Script to build the 4 benchmark planted-leak notebooks (examples/01-04)."""
import os
import nbformat as nbf

os.makedirs("examples", exist_ok=True)

# -------------------------------------------------------------
# Notebook 1: Clear Leak (Hastie & Tibshirani Feature Selection Leak)
# -------------------------------------------------------------
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell(
        "# Example 01: Clear Preprocessing & Feature Selection Leakage (Type 1)\n\n"
        "**Context**: Based on the classic Hastie & Tibshirani textbook scenario (ESL Ch 7.10.2: *'How Not to Do Cross-Validation'*).\n"
        "High-dimensional tabular data with 200 samples and 1,000 features.\n\n"
        "**Planted Flaw**: `StandardScaler` and `SelectKBest` are fitted on the entire dataset `X` "
        "before `train_test_split` is called.\n"
        "**Ground Truth**: `CONFIRMED` Type 1 Leakage.\n"
        "**Expected Outcome**: Leaked AUC ~0.76; Clean/Patched AUC ~0.50 (Delta ~ -0.25)."
    ),
    nbf.v4.new_code_cell(
        "import numpy as np\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.feature_selection import SelectKBest, f_classif\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score, roc_auc_score"
    ),
    nbf.v4.new_code_cell(
        "# Generate 200 samples with 1000 noisy/uninformative features\n"
        "np.random.seed(42)\n"
        "X = np.random.randn(200, 1000)\n"
        "y = np.random.randint(0, 2, size=200)"
    ),
    nbf.v4.new_code_cell(
        "# FLAGGED LEAK: Preprocessing & feature selection fitted on entire dataset before split\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(X)\n"
        "\n"
        "selector = SelectKBest(score_func=f_classif, k=10)\n"
        "X_selected = selector.fit_transform(X_scaled, y)"
    ),
    nbf.v4.new_code_cell(
        "# Split happens AFTER preprocessing and feature selection\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X_selected, y, test_size=0.30, random_state=42\n"
        ")"
    ),
    nbf.v4.new_code_cell(
        "clf = LogisticRegression(random_state=42)\n"
        "clf.fit(X_train, y_train)\n"
        "\n"
        "y_pred = clf.predict(X_test)\n"
        "y_prob = clf.predict_proba(X_test)[:, 1]\n"
        "\n"
        "test_acc = accuracy_score(y_test, y_pred)\n"
        "test_auc = roc_auc_score(y_test, y_prob)\n"
        "print(f\"Test Accuracy: {test_acc:.4f}\")\n"
        "print(f\"Test ROC-AUC: {test_auc:.4f}\")"
    )
]

with open("examples/01_clear_leak_scaler_before_split.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb1, f)

# -------------------------------------------------------------
# Notebook 2: Clean Pipeline (No Leak - Negative Control)
# -------------------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell(
        "# Example 02: Clean Pipeline (No Leak - Negative Control)\n\n"
        "**Clean Structure**: Splitting is executed first. Preprocessing and feature selection "
        "are encapsulated inside an idiomatic scikit-learn `Pipeline` and fitted only on training data.\n"
        "**Ground Truth**: `CLEAN` / `NO_LEAK_DETECTED`."
    ),
    nbf.v4.new_code_cell(
        "import numpy as np\n"
        "from sklearn.datasets import make_classification\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.feature_selection import SelectKBest, f_classif\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score, roc_auc_score"
    ),
    nbf.v4.new_code_cell(
        "# Generate classification dataset\n"
        "np.random.seed(42)\n"
        "X = np.random.randn(200, 100)\n"
        "y = np.random.randint(0, 2, size=200)"
    ),
    nbf.v4.new_code_cell(
        "# CLEAN: Split is performed first, before any fitting occurs\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.30, random_state=42\n"
        ")"
    ),
    nbf.v4.new_code_cell(
        "# CLEAN: Preprocessing & feature selection enclosed in Pipeline\n"
        "pipeline = Pipeline([\n"
        "    ('scaler', StandardScaler()),\n"
        "    ('selector', SelectKBest(score_func=f_classif, k=10)),\n"
        "    ('clf', LogisticRegression(random_state=42))\n"
        "])\n"
        "\n"
        "# Fitting occurs strictly on X_train, y_train\n"
        "pipeline.fit(X_train, y_train)"
    ),
    nbf.v4.new_code_cell(
        "y_pred = pipeline.predict(X_test)\n"
        "y_prob = pipeline.predict_proba(X_test)[:, 1]\n"
        "\n"
        "test_acc = accuracy_score(y_test, y_pred)\n"
        "test_auc = roc_auc_score(y_test, y_prob)\n"
        "print(f\"Clean Pipeline Test Accuracy: {test_acc:.4f}\")\n"
        "print(f\"Clean Pipeline Test ROC-AUC: {test_auc:.4f}\")"
    )
]

with open("examples/02_clean_pipeline_no_leak.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb2, f)

# -------------------------------------------------------------
# Notebook 3: Subtle Leak (Duplicate / Cross-Split Overlap)
# -------------------------------------------------------------
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell(
        "# Example 03: Cross-Split Sample Duplication Leakage (Type 2)\n\n"
        "**Planted Flaw**: Premature duplication / naive oversampling is applied to the full dataset "
        "before splitting, causing identical feature rows to appear in both train and test splits.\n"
        "The labels contain realistic noise (`flip_y=0.2`), so memorised duplicates inflate the score clearly.\n"
        "**Ground Truth**: `CONFIRMED` Type 2 Leakage."
    ),
    nbf.v4.new_code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.datasets import make_classification\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.metrics import accuracy_score, roc_auc_score"
    ),
    nbf.v4.new_code_cell(
        "# Generate base classification dataset\n"
        "np.random.seed(42)\n"
        "X_raw, y_raw = make_classification(\n"
        "    n_samples=200,\n"
        "    n_features=20,\n"
        "    n_informative=10,\n"
        "    flip_y=0.2,\n"
        "    random_state=42\n"
        ")\n"
        "\n"
        "# FLAGGED LEAK: Naive duplication / oversampling applied before split\n"
        "dup_indices = np.random.choice(len(X_raw), size=100, replace=True)\n"
        "X = np.vstack([X_raw, X_raw[dup_indices]])\n"
        "y = np.concatenate([y_raw, y_raw[dup_indices]])"
    ),
    nbf.v4.new_code_cell(
        "# Naive random split places duplicate rows in BOTH train and test\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.25, random_state=42\n"
        ")"
    ),
    nbf.v4.new_code_cell(
        "clf = RandomForestClassifier(n_estimators=50, random_state=42)\n"
        "clf.fit(X_train, y_train)\n"
        "\n"
        "y_pred = clf.predict(X_test)\n"
        "y_prob = clf.predict_proba(X_test)[:, 1]\n"
        "\n"
        "test_acc = accuracy_score(y_test, y_pred)\n"
        "test_auc = roc_auc_score(y_test, y_prob)\n"
        "print(f\"Duplicated Split Test Accuracy: {test_acc:.4f}\")\n"
        "print(f\"Duplicated Split Test ROC-AUC: {test_auc:.4f}\")"
    )
]

with open("examples/03_subtle_leak_duplicate_split.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb3, f)

# -------------------------------------------------------------
# Notebook 4: Test set used for model selection (Type 3)
# -------------------------------------------------------------
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    nbf.v4.new_markdown_cell(
        "# Example 04: Hyperparameter Chosen on the Test Set (Type 3)\n\n"
        "**Planted Flaw**: a loop trains one model per `max_depth` and keeps the one with the best "
        "*test* accuracy. The test set is used for model selection, so the reported score is optimistic.\n"
        "**Ground Truth**: `SUSPECTED` Type 3 leakage (REPROVE has no automatic fix for this type, so it "
        "reports the evidence and does not claim a measured effect)."
    ),
    nbf.v4.new_code_cell(
        "import numpy as np\n"
        "from sklearn.datasets import make_classification\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "from sklearn.metrics import accuracy_score"
    ),
    nbf.v4.new_code_cell(
        "X, y = make_classification(n_samples=300, n_features=20, n_informative=4, flip_y=0.2, random_state=42)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
    ),
    nbf.v4.new_code_cell(
        "# FLAGGED LEAK: every candidate is scored on the TEST set and the best one is kept\n"
        "best_acc, best_depth = 0.0, None\n"
        "for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15, None]:\n"
        "    model = DecisionTreeClassifier(max_depth=depth, random_state=42)\n"
        "    model.fit(X_train, y_train)\n"
        "    acc = accuracy_score(y_test, model.predict(X_test))\n"
        "    if acc > best_acc:\n"
        "        best_acc, best_depth = acc, depth"
    ),
    nbf.v4.new_code_cell(
        "test_acc = best_acc\n"
        "print(f\"Best max_depth={best_depth}, reported test accuracy={test_acc:.4f}\")"
    ),
]

with open("examples/04_test_set_model_selection.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb4, f)

print("Successfully rebuilt all 4 benchmark notebooks in examples/.")
