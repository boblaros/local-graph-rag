"""Pure metrics and paired comparisons; no pipeline implementation lives here."""

from .manifest import MANIFEST_SCHEMA_VERSION, build_analysis_manifest
from .downstream import (
    DEFAULT_PAIRED_METRIC_FIELDS,
    DEFAULT_UNANSWERABLE_TOKEN,
    DOWNSTREAM_METRICS_SCHEMA_VERSION,
    LINEAGE_FIELDS,
    aggregate_downstream_metrics,
    compute_downstream_metrics,
    compute_question_downstream_metrics,
)
from .er_metrics import compute_er_metrics, compute_graph_topology
from .extraction_metrics import (
    compute_extraction_efficiency,
    compute_extraction_metrics,
)
from .paired import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    paired_regime_deltas,
    summarize_regime_deltas,
)
from .primary import (
    CASCADE_COMPARISON_PLAN,
    PRIMARY_ANALYSIS_SCHEMA_VERSION,
    collect_prespecified_cascade_comparisons,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "build_analysis_manifest",
    "compute_er_metrics",
    "compute_extraction_efficiency",
    "compute_extraction_metrics",
    "compute_graph_topology",
    "compute_downstream_metrics",
    "compute_question_downstream_metrics",
    "aggregate_downstream_metrics",
    "DEFAULT_PAIRED_METRIC_FIELDS",
    "DEFAULT_UNANSWERABLE_TOKEN",
    "DOWNSTREAM_METRICS_SCHEMA_VERSION",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "LINEAGE_FIELDS",
    "paired_regime_deltas",
    "summarize_regime_deltas",
    "CASCADE_COMPARISON_PLAN",
    "PRIMARY_ANALYSIS_SCHEMA_VERSION",
    "collect_prespecified_cascade_comparisons",
]
