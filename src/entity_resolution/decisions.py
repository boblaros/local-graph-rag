"""Fixed score-policy and judge-admission decision pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .judge import FixedJudge, JudgeCache, JudgeIdentity
from .models import (
    CandidatePair,
    CannotLink,
    ERConfig,
    EntityProfile,
    PairDecision,
    PairScore,
)


class JudgeBudgetExceededError(RuntimeError):
    """Raised before any judge call when the frozen wall-time budget is exceeded."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(
            "projected ER judge decisions exceed frozen per-run budget: "
            f"{self.report['admitted_ambiguous_pairs']} > "
            f"{self.report['max_judge_calls_per_run']}"
        )


@dataclass(frozen=True)
class DecisionBatch:
    decisions: tuple[PairDecision, ...]
    cannot_links: tuple[CannotLink, ...]
    judge_budget: Mapping[str, Any]


def _judge_admission_set(
    scores: Sequence[PairScore],
    candidates: Mapping[str, CandidatePair],
    config: ERConfig,
) -> tuple[set[str], int]:
    policy = config.decision_policy
    # Judge review requires evidence from the fixed candidate method.
    if not candidates:
        return set(), 0

    route_eligible: list[PairScore] = []
    for score in scores:
        candidate = candidates.get(score.pair_id)
        if (
            candidate is not None
            and score.aggregate_score is not None
            and policy.reject_below_score
            <= score.aggregate_score
            < policy.auto_merge_at_score
            and policy.judge_candidate_method in candidate.methods
        ):
            route_eligible.append(score)

    ranked_by_mention: dict[str, list[PairScore]] = {}
    for score in route_eligible:
        ranked_by_mention.setdefault(score.left_mention_id, []).append(score)
        ranked_by_mention.setdefault(score.right_mention_id, []).append(score)
    top_pairs: dict[str, set[str]] = {}
    for mention_id, mention_scores in ranked_by_mention.items():
        ranked = sorted(
            mention_scores,
            key=lambda item: (-float(item.aggregate_score or 0.0), item.pair_id),
        )
        top_pairs[mention_id] = {
            item.pair_id for item in ranked[: policy.judge_mutual_top_k]
        }
    admitted = {
        score.pair_id
        for score in route_eligible
        if score.pair_id in top_pairs.get(score.left_mention_id, set())
        and score.pair_id in top_pairs.get(score.right_mention_id, set())
    }
    return admitted, len(route_eligible)


def decide_pairs(
    scores: Sequence[PairScore],
    profiles: Sequence[EntityProfile],
    config: ERConfig | None = None,
    *,
    judge: FixedJudge | None = None,
    judge_cache: JudgeCache | None = None,
    candidates: Sequence[CandidatePair] | None = None,
    operational_max_judge_calls_per_run: int | None = None,
) -> DecisionBatch:
    """Apply the fixed score policy and judge admission to candidate pairs."""

    config = config or ERConfig()
    by_id = {profile.mention_id: profile for profile in profiles}
    if judge is not None and (
        not isinstance(getattr(judge, "identity", None), JudgeIdentity)
        or not callable(getattr(judge, "decide", None))
    ):
        raise TypeError("judge must implement FixedJudge with an explicit identity")
    cache = judge_cache or JudgeCache()
    candidate_by_pair = {item.pair_id: item for item in candidates or ()}
    if len(candidate_by_pair) != len(candidates or ()):
        raise ValueError("duplicate candidate pair_id supplied to decision pipeline")
    admitted_pair_ids, route_eligible_ambiguous = _judge_admission_set(
        scores, candidate_by_pair, config
    )
    admitted_ambiguous = len(admitted_pair_ids)
    policy = config.decision_policy
    effective_budget = (
        policy.max_judge_calls_per_run
        if operational_max_judge_calls_per_run is None
        else operational_max_judge_calls_per_run
    )
    if isinstance(effective_budget, bool) or not isinstance(effective_budget, int):
        raise TypeError("operational judge budget must be an integer")
    if effective_budget < policy.max_judge_calls_per_run:
        raise ValueError(
            "operational judge budget cannot reduce the frozen planned budget"
        )
    budget_report = {
        "decision_policy_version": policy.version,
        "admitted_ambiguous_pairs": admitted_ambiguous,
        "route_eligible_ambiguous_pairs": route_eligible_ambiguous,
        "planned_max_judge_calls_per_run": policy.max_judge_calls_per_run,
        "max_judge_calls_per_run": effective_budget,
        "operational_budget_overridden": (
            effective_budget != policy.max_judge_calls_per_run
        ),
        "judge_candidate_method": policy.judge_candidate_method,
        "judge_mutual_top_k": policy.judge_mutual_top_k,
        "passed": admitted_ambiguous <= effective_budget,
    }
    if judge is not None and not budget_report["passed"]:
        raise JudgeBudgetExceededError(budget_report)
    decisions: list[PairDecision] = []
    cannot_links: list[CannotLink] = []

    for score in sorted(scores, key=lambda item: item.pair_id):
        left = by_id[score.left_mention_id]
        right = by_id[score.right_mention_id]
        if score.aggregate_score is None:
            decision = PairDecision(
                pair_id=score.pair_id,
                left_mention_id=left.mention_id,
                right_mention_id=right.mention_id,
                action="abstain",
                source="no_judge",
                score=None,
                rationale="no available weighted signals",
            )
        elif score.aggregate_score >= policy.auto_merge_at_score:
            decision = PairDecision(
                pair_id=score.pair_id,
                left_mention_id=left.mention_id,
                right_mention_id=right.mention_id,
                action="merge",
                source="score_policy_merge",
                score=score.aggregate_score,
                rationale=(
                    "score >= auto-merge boundary "
                    f"({policy.auto_merge_at_score:.6f})"
                ),
            )
        elif score.aggregate_score < policy.reject_below_score:
            decision = PairDecision(
                pair_id=score.pair_id,
                left_mention_id=left.mention_id,
                right_mention_id=right.mention_id,
                action="reject",
                source="score_policy_reject",
                score=score.aggregate_score,
                rationale=(
                    "score < reject boundary "
                    f"({policy.reject_below_score:.6f})"
                ),
            )
        elif score.pair_id not in admitted_pair_ids:
            decision = PairDecision(
                pair_id=score.pair_id,
                left_mention_id=left.mention_id,
                right_mention_id=right.mention_id,
                action="abstain",
                source="judge_admission",
                score=score.aggregate_score,
                rationale=(
                    "ambiguous score lacks a frozen identity-bearing judge "
                    "admission route"
                ),
            )
        elif judge is None:
            decision = PairDecision(
                pair_id=score.pair_id,
                left_mention_id=left.mention_id,
                right_mention_id=right.mention_id,
                action="abstain",
                source="no_judge",
                score=score.aggregate_score,
                rationale="ambiguous score and no fixed judge supplied",
            )
        else:
            result, cache_key, cache_hit = cache.resolve(judge, left, right, score)
            decision = PairDecision(
                pair_id=score.pair_id,
                left_mention_id=left.mention_id,
                right_mention_id=right.mention_id,
                action=result.action,
                source="judge",
                score=score.aggregate_score,
                rationale=result.rationale,
                judge_cache_key=cache_key,
                judge_cache_hit=cache_hit,
                judge_metadata={
                    **result.metadata,
                    "judge_relationship": result.relationship,
                    "judge_tag": judge.identity.model_tag,
                    "judge_digest": judge.identity.model_digest,
                    "judge_prompt_version": judge.identity.prompt_version,
                    "judge_config_hash": judge.identity.config_hash,
                },
            )
        decisions.append(decision)
        if decision.action == "reject":
            cannot_links.append(
                CannotLink(
                    left_mention_id=left.mention_id,
                    right_mention_id=right.mention_id,
                    reason=decision.rationale,
                    pair_id=decision.pair_id,
                    source=decision.source,
                )
            )

    deduplicated = {link.key: link for link in cannot_links}
    return DecisionBatch(
        decisions=tuple(decisions),
        cannot_links=tuple(deduplicated[key] for key in sorted(deduplicated)),
        judge_budget=budget_report,
    )
