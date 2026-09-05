"""Deterministic merge-only clustering over immutable Native entities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .models import (
    CannotLink,
    Cluster,
    EntityProfile,
    PairDecision,
    stable_id,
)


@dataclass(frozen=True)
class BlockedMerge:
    trigger_pair_id: str
    left_cluster_members: tuple[str, ...]
    right_cluster_members: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClusterResult:
    clusters: tuple[Cluster, ...]
    blocked_merges: tuple[BlockedMerge, ...]


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def constrained_cluster(
    profiles: Sequence[EntityProfile],
    decisions: Sequence[PairDecision],
    *,
    cannot_links: Sequence[CannotLink] = (),
) -> ClusterResult:
    """Union accepted merge edges unless an explicit cannot-link vetoes them.

    Abstentions and missing pairs are deliberately neutral.  A transitive merge
    is blocked only when at least one explicit reject-derived cannot-link would
    end up inside the proposed cluster.
    """
    by_id = {profile.mention_id: profile for profile in profiles}
    if len(by_id) != len(profiles):
        raise ValueError("duplicate mention_id in profiles")
    decision_by_key: dict[tuple[str, str], PairDecision] = {}
    for decision in decisions:
        key = _pair_key(decision.left_mention_id, decision.right_mention_id)
        if key in decision_by_key:
            raise ValueError(f"duplicate pair decision for {key}")
        if key[0] not in by_id or key[1] not in by_id:
            raise ValueError(f"pair decision references unknown mention: {key}")
        decision_by_key[key] = decision

    cannot_keys: set[tuple[str, str]] = set()
    for link in cannot_links:
        if link.left_mention_id not in by_id or link.right_mention_id not in by_id:
            raise ValueError(f"cannot-link references unknown mention: {link.key}")
        cannot_keys.add(link.key)
    clusters: dict[str, set[str]] = {mention_id: {mention_id} for mention_id in by_id}
    member_root: dict[str, str] = {mention_id: mention_id for mention_id in by_id}
    merge_evidence: dict[str, set[str]] = {mention_id: set() for mention_id in by_id}
    blocked: list[BlockedMerge] = []

    merge_decisions = sorted(
        (decision for decision in decisions if decision.action == "merge"),
        key=lambda item: (
            -(item.score if item.score is not None else -1.0),
            item.pair_id,
        ),
    )
    for decision in merge_decisions:
        left_root = member_root[decision.left_mention_id]
        right_root = member_root[decision.right_mention_id]
        if left_root == right_root:
            continue
        left_members = clusters[left_root]
        right_members = clusters[right_root]
        reason: str | None = None
        cross_pairs = [
            _pair_key(left, right)
            for left in sorted(left_members)
            for right in sorted(right_members)
        ]
        if reason is None:
            contradiction = next(
                (key for key in cross_pairs if key in cannot_keys), None
            )
            if contradiction is not None:
                reason = f"cannot_link:{contradiction[0]}:{contradiction[1]}"
        if reason is not None:
            blocked.append(
                BlockedMerge(
                    trigger_pair_id=decision.pair_id,
                    left_cluster_members=tuple(sorted(left_members)),
                    right_cluster_members=tuple(sorted(right_members)),
                    reason=reason,
                )
            )
            continue

        combined = left_members | right_members
        new_root = min(combined)
        new_evidence = (
            merge_evidence[left_root]
            | merge_evidence[right_root]
            | {
                pair_decision.pair_id
                for key in cross_pairs
                if (pair_decision := decision_by_key.get(key)) is not None
                and pair_decision.action == "merge"
            }
        )
        del clusters[left_root]
        del clusters[right_root]
        del merge_evidence[left_root]
        del merge_evidence[right_root]
        clusters[new_root] = combined
        merge_evidence[new_root] = new_evidence
        for member in combined:
            member_root[member] = new_root

    final = tuple(
        sorted(
            (
                Cluster(
                    cluster_id=stable_id("cluster_", sorted(members)),
                    mention_ids=tuple(members),
                    merge_evidence_pair_ids=tuple(merge_evidence[root]),
                )
                for root, members in clusters.items()
            ),
            key=lambda item: item.mention_ids,
        )
    )
    return ClusterResult(clusters=final, blocked_merges=tuple(blocked))
