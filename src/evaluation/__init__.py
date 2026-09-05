"""Pure metrics and paired comparisons; no pipeline implementation lives here."""

from .downstream import (
    DEFAULT_PAIRED_METRIC_FIELDS,
    DEFAULT_RETRIEVAL_CUTOFFS,
    DEFAULT_UNANSWERABLE_TOKEN,
    DOWNSTREAM_METRICS_SCHEMA_VERSION,
    LINEAGE_FIELDS,
    SOURCE_QUESTION_TYPE_WEIGHTS,
    aggregate_downstream_metrics,
    compute_downstream_metrics,
    compute_question_downstream_metrics,
)
from .er_metrics import compute_er_metrics, compute_graph_topology
from .exploratory import (
    EXPLORATORY_BINARY_METRICS,
    EXPLORATORY_SCHEMA_VERSION,
    builder_metadata_from_config,
    compute_exploratory_analysis,
    exact_mcnemar,
    holm_adjust,
    summarize_builder_artifacts,
)
from .extraction_metrics import (
    bootstrap_extraction_metrics,
    compute_extraction_efficiency,
    compute_extraction_metrics,
)
from .paired import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    paired_variant_deltas,
    paired_regime_deltas,
    summarize_paired_deltas,
    summarize_regime_deltas,
)
from .primary import (
    CASCADE_COMPARISON_PLAN,
    PRIMARY_ANALYSIS_SCHEMA_VERSION,
    collect_prespecified_cascade_comparisons,
)

__all__ = [
    "compute_er_metrics",
    "EXPLORATORY_BINARY_METRICS",
    "EXPLORATORY_SCHEMA_VERSION",
    "builder_metadata_from_config",
    "compute_exploratory_analysis",
    "exact_mcnemar",
    "holm_adjust",
    "summarize_builder_artifacts",
    "bootstrap_extraction_metrics",
    "compute_extraction_efficiency",
    "compute_extraction_metrics",
    "compute_graph_topology",
    "compute_downstream_metrics",
    "compute_question_downstream_metrics",
    "aggregate_downstream_metrics",
    "DEFAULT_PAIRED_METRIC_FIELDS",
    "DEFAULT_RETRIEVAL_CUTOFFS",
    "DEFAULT_UNANSWERABLE_TOKEN",
    "DOWNSTREAM_METRICS_SCHEMA_VERSION",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "LINEAGE_FIELDS",
    "SOURCE_QUESTION_TYPE_WEIGHTS",
    "paired_variant_deltas",
    "paired_regime_deltas",
    "summarize_paired_deltas",
    "summarize_regime_deltas",
    "CASCADE_COMPARISON_PLAN",
    "PRIMARY_ANALYSIS_SCHEMA_VERSION",
    "collect_prespecified_cascade_comparisons",
]
