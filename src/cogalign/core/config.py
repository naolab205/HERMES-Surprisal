from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _DEFAULT_PROJECT_ROOT / "data_cache"
_DEFAULT_RESULTS_DIR = _DEFAULT_PROJECT_ROOT / "results"

# Supported main-task types (Requirement 7.1, 7.3). ``pair_control`` keeps the
# existing exact-row EEG/Eye-to-Text pair/retrieval behaviour (a strict
# negative/control task); the other three are the new cognitive-language
# objectives that consume EEG/Eye-derived embeddings only.
TASK_TYPE_PAIR_CONTROL = "pair_control"
TASK_TYPE_LOAD_CLASSIFICATION = "semantic_load_classification"
TASK_TYPE_SHIFT_REGRESSION = "semantic_shift_regression"
TASK_TYPE_ALIGNMENT_ANALYSIS = "alignment_analysis"
TASK_TYPE_ANCHOR_RETRIEVAL = "anchor_retrieval"
TASK_TYPES = (
    TASK_TYPE_PAIR_CONTROL,
    TASK_TYPE_LOAD_CLASSIFICATION,
    TASK_TYPE_SHIFT_REGRESSION,
    TASK_TYPE_ALIGNMENT_ANALYSIS,
    TASK_TYPE_ANCHOR_RETRIEVAL,
)

# Default validation early-stop metric per task type (Requirement 6.3). For the
# classification task we monitor a cognitive-load validation metric on the
# EEG/Eye/fused head; for regression we monitor the rank correlation of the
# semantic-shift head; for pair/alignment we keep the legacy pair ROC-AUC.
DEFAULT_EARLY_STOP_METRIC = {
    TASK_TYPE_PAIR_CONTROL: "val_roc_auc",
    TASK_TYPE_LOAD_CLASSIFICATION: "val_cogload_macro_f1",
    TASK_TYPE_SHIFT_REGRESSION: "val_shift_spearman",
    TASK_TYPE_ALIGNMENT_ANALYSIS: "val_roc_auc",
    TASK_TYPE_ANCHOR_RETRIEVAL: "val_anchor_map",
}


def default_early_stop_metric(task_type: str) -> str:
    """Return the default validation early-stop metric for a task type."""
    return DEFAULT_EARLY_STOP_METRIC.get(task_type, "val_roc_auc")


# ---------------------------------------------------------------------------
# Prediction TARGET selector (Requirement 2.1, 5.2, 5.3).
# ---------------------------------------------------------------------------
# The original cognitive-language target is sentence-level contextual
# ``semantic_shift`` (cosine distance of the current BERT row embedding to its
# preceding context). Empirically that signal is near chance, so a stronger,
# standard reading-cognition target is added: ``surprisal`` -- the mean token
# negative log-probability of each row's text under a Chinese causal language
# model (with a BERT pseudo-log-likelihood proxy as an offline fallback).
#
# The target is a LABEL ONLY: it is never a model input. Both targets reuse the
# same train-only tercile machinery (labels.py) and the same regression head.
TARGET_SEMANTIC_SHIFT = "semantic_shift"
TARGET_SURPRISAL = "surprisal"
TARGETS = (TARGET_SEMANTIC_SHIFT, TARGET_SURPRISAL)

# Optimization v2 target family. ``single`` preserves the Iteration-1 scalar
# target. ``surprisal_decomposed`` exposes multiple text-derived label columns
# for mechanism analyses; these columns remain labels/analysis targets only and
# never enter the EEG/Eye encoder.
TARGET_FAMILY_SINGLE = "single"
TARGET_FAMILY_SURPRISAL_DECOMPOSED = "surprisal_decomposed"
TARGET_FAMILIES = (TARGET_FAMILY_SINGLE, TARGET_FAMILY_SURPRISAL_DECOMPOSED)

DECOMPOSED_TARGET_SURPRISAL_MEAN = "surprisal_mean"
DECOMPOSED_TARGET_CONTEXTUAL_ENTROPY = "contextual_entropy"
DECOMPOSED_TARGET_SURPRISAL_DELTA = "surprisal_delta"
DECOMPOSED_TARGET_SHALLOW_SURPRISAL = "shallow_surprisal"
DECOMPOSED_TARGET_DEEP_DISCREPANCY = "deep_discrepancy"
DECOMPOSED_TARGETS = (
    DECOMPOSED_TARGET_SURPRISAL_MEAN,
    DECOMPOSED_TARGET_CONTEXTUAL_ENTROPY,
    DECOMPOSED_TARGET_SURPRISAL_DELTA,
    DECOMPOSED_TARGET_SHALLOW_SURPRISAL,
    DECOMPOSED_TARGET_DEEP_DISCREPANCY,
)

FUSION_GATE = "gate"
FUSION_CONCAT = "concat"
FUSION_RESIDUAL_GATE = "residual_gate"
FUSION_LATE_MOE = "late_moe"
FUSION_UNCERTAINTY_GATE = "uncertainty_gate"
FUSION_MODES = (
    FUSION_GATE,
    FUSION_CONCAT,
    FUSION_RESIDUAL_GATE,
    FUSION_LATE_MOE,
    FUSION_UNCERTAINTY_GATE,
)

EEG_WINDOW_SCHEMA_NONE = "none"
EEG_WINDOW_SCHEMA_N400_P600 = "n400_p600"
EEG_WINDOW_SCHEMAS = (EEG_WINDOW_SCHEMA_NONE, EEG_WINDOW_SCHEMA_N400_P600)

FUSION_FALLBACK_NONE = "none"
FUSION_FALLBACK_EEG_ON_VAL_DROP = "eeg_on_val_drop"
FUSION_FALLBACKS = (FUSION_FALLBACK_NONE, FUSION_FALLBACK_EEG_ON_VAL_DROP)

# Default causal LM and offline proxy for the surprisal label (surprisal.py).
DEFAULT_SURPRISAL_LM = "uer/gpt2-chinese-cluecorpussmall"
DEFAULT_SURPRISAL_PROXY_LM = "bert-base-chinese"
DEFAULT_SURPRISAL_CONTEXT_TOKENS = 192


@dataclass(frozen=True)
class Paths:
    project_root: Path = _DEFAULT_PROJECT_ROOT
    data_root: Path = Path("/data/my/ds004952")
    legacy_epoch_cache: Path = Path("__unused_legacy_epoch_cache__")
    cache_dir: Path = _DEFAULT_CACHE_DIR
    results_dir: Path = _DEFAULT_RESULTS_DIR

    def __post_init__(self) -> None:
        project_root = Path(self.project_root)
        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "data_root", Path(self.data_root))
        if Path(self.cache_dir) == _DEFAULT_CACHE_DIR:
            object.__setattr__(self, "cache_dir", project_root / "data_cache")
        else:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if Path(self.results_dir) == _DEFAULT_RESULTS_DIR:
            object.__setattr__(self, "results_dir", project_root / "results")
        else:
            object.__setattr__(self, "results_dir", Path(self.results_dir))

    @property
    def table_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def fig_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def log_dir(self) -> Path:
        return self.results_dir / "logs"

    @property
    def model_dir(self) -> Path:
        return self.results_dir / "models"

    @property
    def eye_run_cache(self) -> Path:
        return self.cache_dir / "eye_runs"

    @property
    def eye_phase_cache(self) -> Path:
        return self.cache_dir / "eye_phase_runs"

    @property
    def epoch_cache(self) -> Path:
        return self.cache_dir / "epochs"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    fold: int = 0
    n_blocks: int = 7
    subjects: tuple[str, ...] = (
        "sub-04",
        "sub-05",
        "sub-06",
        "sub-07",
        "sub-08",
        "sub-09",
        "sub-10",
        "sub-13",
        "sub-14",
        "sub-15",
    )
    books: tuple[str, ...] = ("LittlePrince", "GarnettDream")
    embed_dim: int = 128
    eeg_channels: int = 128
    eeg_time: int = 128
    text_dim: int = 768
    batch_size: int = 64
    epochs_stage1: int = 6
    epochs_stage2: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.10
    lambda_eeg: float = 0.5
    lambda_eye: float = 0.5
    lambda_fused: float = 1.0
    lambda_pair: float = 1.0
    lambda_cog: float = 0.4
    lambda_subject: float = 0.05
    subject_grl_lambda: float = 0.2
    # Phase 4 (model extension) toggles. Defaults preserve existing behaviour:
    # the optional semantic-shift regression and language-alignment objectives
    # are off unless explicitly enabled.
    n_load_classes: int = 3
    use_semantic_shift: bool = False
    use_alignment: bool = False
    alignment_use_cka: bool = False
    lambda_align: float = 0.0
    lambda_shift: float = 0.0
    use_aux_shift: bool = False
    lambda_aux_shift: float = 0.0
    aux_shift_loss_branches: tuple[str, ...] = ("fused",)
    # Phase 5 (training script) task-type selector. ``pair_control`` preserves
    # the existing exact-row pair/retrieval objective; the cognitive-language
    # task types switch the primary objective wired in ``train_model``.
    task_type: str = TASK_TYPE_PAIR_CONTROL
    # Prediction target selector. ``semantic_shift`` preserves the original
    # contextual-shift label; ``surprisal`` switches the regression/tercile
    # target to LM surprisal (surprisal.py). The target is a LABEL ONLY.
    target: str = TARGET_SEMANTIC_SHIFT
    target_family: str = TARGET_FAMILY_SINGLE
    target_component: str = DECOMPOSED_TARGET_SURPRISAL_MEAN
    surprisal_lm: str = DEFAULT_SURPRISAL_LM
    surprisal_proxy_lm: str = DEFAULT_SURPRISAL_PROXY_LM
    surprisal_context_tokens: int = DEFAULT_SURPRISAL_CONTEXT_TOKENS
    tercile_strategy: str = "tercile"
    max_items_per_book: int | None = None
    max_samples_per_split: int | None = None
    context_window: int = 3
    eye_source: str = "rar"
    num_workers: int = 0
    bootstrap_iters: int = 200
    retrieval_pools: tuple[int, ...] = (10, 50, 100)
    device: str = "cpu"
    use_eeg: bool = True
    use_eye: bool = True
    # "gate" = legacy convex gate, "concat" = existing EEG-residual concat,
    # "residual_gate" = Optimization-v2 do-no-harm residual eye correction,
    # "late_moe" = train branches normally and select/weight scalar readout on
    # validation only in the evaluation layer.
    fusion: str = FUSION_RESIDUAL_GATE
    eeg_window_schema: str = EEG_WINDOW_SCHEMA_NONE
    fusion_fallback: str = FUSION_FALLBACK_EEG_ON_VAL_DROP
    fusion_min_delta: float = 0.005
    shift_loss_branches: tuple[str, ...] = ("fused",)
    anchor_vocab_size: int = 512
    anchors_per_row: int = 3
    lambda_anchor: float = 1.0
    lambda_retrieval: float = 1.0
    prediction_head: str = "linear"
    protocol: str = "blocked"
    test_subject: str | None = None
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4
    early_stop_metric: str = "val_roc_auc"
    run_name: str = "cogalign"
    paths: Paths = field(default_factory=Paths)


BOOK_TO_SESSION = {
    "LittlePrince": "ses-LittlePrince",
    "GarnettDream": "ses-GarnettDream",
}

SESSION_TO_BOOK = {v: k for k, v in BOOK_TO_SESSION.items()}

EYE_FEATURE_NAMES = (
    "fixation_rate",
    "mean_fixation_duration",
    "saccade_rate",
    "regression_rate",
    "gaze_dispersion",
    "mean_pupil",
    "pupil_slope",
    "pupil_variability",
    "valid_ratio",
)

FORBIDDEN_MODEL_COLUMNS = (
    "subject",
    "book",
    "session",
    "run",
    "row_idx",
    "item_id",
    "text",
    "n_char",
    "dur",
    "raw_duration",
    "n_samples",
    # Optimization v2 decomposed linguistic targets. These are labels/analysis
    # columns only; none may enter the EEG/Eye encoder.
    "surprisal",
    "surprisal_mean",
    "contextual_entropy",
    "surprisal_delta",
    "shallow_surprisal",
    "deep_discrepancy",
    "target_valid",
    # Confound fields (CogAlign v2): may only feed the confound-residual branch,
    # never the neural encoder (leakage-audit item E).
    "position",
    "fixation_count",
)
