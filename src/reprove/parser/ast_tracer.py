"""AST Tracer that extracts structured Experiment Graphs from Python code."""
import ast
from typing import Dict, List, Optional, Set, Tuple

from reprove.parser.models import CodeSpan, ExperimentGraph, PipelineNode, StageType
from reprove.parser.notebook import NotebookLineMapping


KNOWN_TRANSFORMERS = {
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "Normalizer",
    "MaxAbsScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "LabelEncoder",
    "SelectKBest",
    "SelectPercentile",
    "RFE",
    "RFECV",
    "SelectFromModel",
    "PCA",
    "TruncatedSVD",
    "KernelPCA",
    "PolynomialFeatures",
    "CountVectorizer",
    "TfidfVectorizer",
    "RandomOverSampler",
    "SMOTE",
}

KNOWN_ESTIMATORS = {
    "LogisticRegression",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "SVC",
    "SVR",
    "LinearSVC",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "GaussianNB",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "Pipeline",
    "GridSearchCV",
    "RandomizedSearchCV",
}

KNOWN_DATA_LOADERS = {
    "read_csv",
    "read_parquet",
    "read_table",
    "read_excel",
    "make_classification",
    "make_regression",
    "load_breast_cancer",
    "load_iris",
    "load_digits",
    "load_wine",
    "load_diabetes",
}

KNOWN_METRICS = {
    "accuracy_score",
    "roc_auc_score",
    "f1_score",
    "precision_score",
    "recall_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "classification_report",
    "confusion_matrix",
}


def _extract_name(node: ast.AST) -> Optional[str]:
    """Extract a simple string representation from Name or Attribute."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value_name = _extract_name(node.value)
        if value_name:
            return f"{value_name}.{node.attr}"
        return node.attr
    return None


def _get_call_info(node: ast.AST) -> Tuple[Optional[str], Optional[str]]:
    """Return (caller, function_name) from an ast.Call node."""
    if not isinstance(node, ast.Call):
        return None, None

    func = node.func
    if isinstance(func, ast.Name):
        return None, func.id
    if isinstance(func, ast.Attribute):
        caller = _extract_name(func.value)
        return caller, func.attr
    return None, None


class VariableUsageCollector(ast.NodeVisitor):
    """Collects all variable names read inside an AST sub-tree."""
    def __init__(self):
        self.names: Set[str] = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)
        self.generic_visit(node)


def _collect_read_variables(node: ast.AST) -> List[str]:
    collector = VariableUsageCollector()
    collector.visit(node)
    return sorted(list(collector.names))


def _extract_target_names(target: ast.AST) -> List[str]:
    """Extract assigned variable names from an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_extract_target_names(elt))
        return names
    return []


class ExperimentASTVisitor(ast.NodeVisitor):
    """Walks the AST and builds pipeline nodes representing stages and operations."""

    def __init__(
        self,
        source_code: str,
        file_path: str,
        notebook_mapping: Optional[NotebookLineMapping] = None,
    ):
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.file_path = file_path
        self.notebook_mapping = notebook_mapping

        self.nodes: List[PipelineNode] = []
        self.node_counter = 0

        # Symbol tables to track role of variables
        self.transformer_vars: Set[str] = set()
        self.estimator_vars: Set[str] = set()
        self.var_classes: Dict[str, str] = {}

    def _next_node_id(self) -> str:
        self.node_counter += 1
        return f"node_{self.node_counter}"

    def _get_code_snippet(self, start_line: int, end_line: int) -> str:
        """Extract lines of code (1-indexed inclusive)."""
        idx_start = max(0, start_line - 1)
        idx_end = min(len(self.lines), end_line)
        return "\n".join(self.lines[idx_start:idx_end])

    def _create_code_span(self, stmt: ast.stmt) -> CodeSpan:
        start_line = stmt.lineno
        end_line = getattr(stmt, "end_lineno", stmt.lineno)
        snippet = self._get_code_snippet(start_line, end_line)

        cell_index = None
        cell_line = None
        if self.notebook_mapping:
            loc = self.notebook_mapping.get_notebook_loc(start_line)
            if loc:
                cell_index, cell_line = loc

        return CodeSpan(
            file_path=self.file_path,
            start_line=start_line,
            end_line=end_line,
            cell_index=cell_index,
            cell_line=cell_line,
            code_snippet=snippet,
        )

    def _classify_call(
        self, caller: Optional[str], func_name: Optional[str], call_node: ast.Call
    ) -> StageType:
        """Classify a function/method call into an experiment lifecycle stage."""
        if not func_name:
            return StageType.OTHER

        # 1. Splitting
        if (
            func_name == "train_test_split"
            or "KFold" in func_name
            or "Split" in func_name
        ):
            return StageType.SPLITTING

        # 2. Data loading
        if (
            func_name in KNOWN_DATA_LOADERS
            or func_name.startswith("read_")
            or func_name.startswith("load_")
        ):
            return StageType.DATA_LOADING

        # 3. Model Training
        if func_name == "fit":
            # Check if caller is known transformer or estimator
            if caller and caller in self.transformer_vars:
                return StageType.PREPROCESSING
            if caller and caller in self.estimator_vars:
                return StageType.MODEL_TRAINING
            # Default fit to MODEL_TRAINING if ambiguous
            return StageType.MODEL_TRAINING

        # 4. Preprocessing transforms
        if func_name in {"fit_transform", "transform", "fit_resample"}:
            return StageType.PREPROCESSING

        # 5. Instantiations of transformers / estimators
        if func_name in KNOWN_TRANSFORMERS:
            return StageType.PREPROCESSING
        if func_name in KNOWN_ESTIMATORS:
            return StageType.MODEL_TRAINING

        # 6. Evaluation
        if (
            func_name in KNOWN_METRICS
            or func_name in {"predict", "predict_proba", "score"}
            or "score" in func_name.lower()
        ):
            return StageType.EVALUATION

        return StageType.OTHER

    def visit_Assign(self, node: ast.Assign):
        self._process_assignment(node, node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value:
            self._process_assignment(node, [node.target], node.value)
        self.generic_visit(node)

    def _process_assignment(
        self, stmt: ast.stmt, targets: List[ast.AST], value: ast.AST
    ):
        outputs: List[str] = []
        for target in targets:
            outputs.extend(_extract_target_names(target))

        inputs = _collect_read_variables(value)

        # Check if right-hand side is a Call
        if isinstance(value, ast.Call):
            caller, func_name = _get_call_info(value)
            stage = self._classify_call(caller, func_name, value)

            # Record symbol tracking
            if func_name in KNOWN_TRANSFORMERS:
                for out_var in outputs:
                    self.transformer_vars.add(out_var)
                    self.var_classes[out_var] = func_name
            elif func_name in KNOWN_ESTIMATORS:
                for out_var in outputs:
                    self.estimator_vars.add(out_var)
                    self.var_classes[out_var] = func_name

            caller_class = self.var_classes.get(caller) if caller else None

            # Label representation
            out_str = ", ".join(outputs) if outputs else "_"
            label = f"{out_str} = {caller + '.' if caller else ''}{func_name}(...)"

            self.nodes.append(
                PipelineNode(
                    node_id=self._next_node_id(),
                    label=label,
                    stage=stage,
                    code_span=self._create_code_span(stmt),
                    caller=caller,
                    caller_class=caller_class,
                    function_name=func_name,
                    inputs=inputs,
                    outputs=outputs,
                )
            )
        else:
            # Simple assignment / operation (e.g. data concatenation or slicing)
            code_snippet = self._get_code_snippet(stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno))
            stage = StageType.OTHER
            # Check for duplication / concatenation operations
            if "vstack" in code_snippet or "concat" in code_snippet or "choice" in code_snippet:
                stage = StageType.PREPROCESSING

            out_str = ", ".join(outputs) if outputs else "_"
            self.nodes.append(
                PipelineNode(
                    node_id=self._next_node_id(),
                    label=f"{out_str} = <expression>",
                    stage=stage,
                    code_span=self._create_code_span(stmt),
                    inputs=inputs,
                    outputs=outputs,
                )
            )

    def visit_Expr(self, node: ast.Expr):
        """Handle standalone calls (e.g. clf.fit(X, y) or print(...))."""
        if isinstance(node.value, ast.Call):
            caller, func_name = _get_call_info(node.value)
            caller_class = self.var_classes.get(caller) if caller else None
            inputs = _collect_read_variables(node.value)
            stage = self._classify_call(caller, func_name, node.value)

            label = f"{caller + '.' if caller else ''}{func_name}(...)"
            self.nodes.append(
                PipelineNode(
                    node_id=self._next_node_id(),
                    label=label,
                    stage=stage,
                    code_span=self._create_code_span(node),
                    caller=caller,
                    caller_class=caller_class,
                    function_name=func_name,
                    inputs=inputs,
                    outputs=[],
                )
            )
        self.generic_visit(node)


def build_experiment_graph(
    source_code: str,
    file_path: str,
    notebook_mapping: Optional[NotebookLineMapping] = None,
) -> ExperimentGraph:
    """Parse source code with AST and build a line-indexed ExperimentGraph."""
    parsed_ast = ast.parse(source_code, filename=file_path)
    visitor = ExperimentASTVisitor(source_code, file_path, notebook_mapping)
    visitor.visit(parsed_ast)

    return ExperimentGraph(
        source_file=file_path,
        raw_code=source_code,
        nodes=visitor.nodes,
    )
