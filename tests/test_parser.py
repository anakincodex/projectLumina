"""Unit tests for AST Parser and Experiment Graph generator."""
import pytest
from reprove.parser import parse_file, StageType, build_experiment_graph


def test_parse_01_clear_leak_notebook():
    """Verify that the parser correctly extracts the sequence where fit_transform precedes split."""
    graph = parse_file("examples/01_clear_leak_scaler_before_split.ipynb")
    assert graph is not None
    assert len(graph.nodes) > 0

    splits = graph.find_splits()
    assert len(splits) == 1, f"Expected exactly 1 split node, found {len(splits)}"
    split_node = splits[0]
    assert split_node.function_name == "train_test_split"
    assert "X_train" in split_node.outputs
    assert split_node.code_span.cell_index is not None

    fits = graph.find_fits()
    assert len(fits) >= 2  # scaler.fit_transform, selector.fit_transform, and clf.fit

    # Preprocessing fit_transform nodes must appear BEFORE the split node
    prep_fits = [f for f in fits if f.stage == StageType.PREPROCESSING]
    assert len(prep_fits) >= 2, "Expected at least 2 preprocessing fit_transform nodes"

    for prep_fit in prep_fits:
        assert prep_fit.code_span.start_line < split_node.code_span.start_line, (
            f"Expected preprocessing node ({prep_fit.label}) at L{prep_fit.code_span.start_line} "
            f"to precede split at L{split_node.code_span.start_line}"
        )


def test_parse_02_clean_pipeline_notebook():
    """Verify that in the clean notebook, split occurs BEFORE model training fit."""
    graph = parse_file("examples/02_clean_pipeline_no_leak.ipynb")
    assert graph is not None

    splits = graph.find_splits()
    assert len(splits) == 1
    split_node = splits[0]

    fits = graph.find_fits()
    # In clean pipeline, the fit is on the Pipeline estimator, which occurs after split
    assert len(fits) == 1
    model_fit = fits[0]
    assert model_fit.stage == StageType.MODEL_TRAINING
    assert model_fit.caller == "pipeline"
    assert split_node.code_span.start_line < model_fit.code_span.start_line


def test_parse_03_subtle_leak_notebook():
    """Verify that in the subtle leak notebook, duplication operations and split are captured."""
    graph = parse_file("examples/03_subtle_leak_duplicate_split.ipynb")
    assert graph is not None

    splits = graph.find_splits()
    assert len(splits) == 1

    # Check that random forest training is classified as MODEL_TRAINING
    model_nodes = graph.get_nodes_by_stage(StageType.MODEL_TRAINING)
    assert len(model_nodes) >= 1


def test_parser_handles_ipython_magics(tmp_path):
    """Verify that IPython magics (%matplotlib, !pip) do not break parsing."""
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_code_cell(
            "%matplotlib inline\n"
            "!pip install scikit-learn\n"
            "from sklearn.model_selection import train_test_split\n"
            "X_tr, X_te = train_test_split([1, 2, 3, 4])"
        )
    ]
    nb_path = str(tmp_path / "test_magic.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    graph = parse_file(nb_path)
    assert len(graph.find_splits()) == 1
    split_node = graph.find_splits()[0]
    assert split_node.code_span.cell_index == 0


def test_parse_python_script(tmp_path):
    """Verify direct parsing of a plain .py file with exact line numbers."""
    py_code = (
        "import numpy as np\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "X = np.ones((10, 5))\n"
        "scaler = StandardScaler()\n"
        "X_s = scaler.fit_transform(X)\n"
        "X_tr, X_te = train_test_split(X_s, test_size=0.2)\n"
    )
    py_file = tmp_path / "train.py"
    py_file.write_text(py_code, encoding="utf-8")

    graph = parse_file(str(py_file))
    assert len(graph.nodes) > 0
    splits = graph.find_splits()
    assert len(splits) == 1
    assert splits[0].code_span.start_line == 7
