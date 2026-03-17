from collections import defaultdict

import numpy as np


class StratifiedGroupKFoldNER:
    """Group-aware NER splitter with non-empty folds and local-search refinement."""

    def __init__(self, n_splits, seed=42):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.seed = seed
        self.last_summary = None

    def _build_profile(self, group_id, indices, dataset, label_to_idx, num_labels):
        presence = np.zeros(num_labels, dtype=float)
        span_counts = np.zeros(num_labels, dtype=float)
        for index in indices:
            for _start, _end, label in dataset[index].get("ner", []):
                label_idx = label_to_idx[label]
                presence[label_idx] = 1.0
                span_counts[label_idx] += 1.0
        return {
            "group_id": group_id,
            "indices": indices,
            "presence": presence,
            "span_counts": span_counts,
            "num_examples": len(indices),
            "num_spans": int(span_counts.sum()),
        }

    def _empty_fold(self, num_labels):
        return {
            "group_ids": [],
            "indices": [],
            "presence": np.zeros(num_labels, dtype=float),
            "span_counts": np.zeros(num_labels, dtype=float),
            "group_count": 0,
            "example_count": 0,
        }

    def _group_capacity_limits(self, num_groups):
        base = num_groups // self.n_splits
        remainder = num_groups % self.n_splits
        return [base + (1 if fold_idx < remainder else 0) for fold_idx in range(self.n_splits)]

    def _add_profile(self, fold, profile):
        fold["group_ids"].append(profile["group_id"])
        fold["indices"].extend(profile["indices"])
        fold["presence"] += profile["presence"]
        fold["span_counts"] += profile["span_counts"]
        fold["group_count"] += 1
        fold["example_count"] += profile["num_examples"]

    def _remove_profile(self, fold, profile):
        fold["group_ids"].remove(profile["group_id"])
        for index in profile["indices"]:
            fold["indices"].remove(index)
        fold["presence"] -= profile["presence"]
        fold["span_counts"] -= profile["span_counts"]
        fold["group_count"] -= 1
        fold["example_count"] -= profile["num_examples"]

    def _cost_components(self, folds, target_groups, target_examples, target_presence, target_spans):
        num_labels = len(target_presence)
        group_cost = 0.0
        example_cost = 0.0
        empty_fold_penalty = 0.0
        missing_label_penalty = 0.0
        presence_cost = 0.0
        span_cost = 0.0

        valid_presence = target_presence > 0
        valid_spans = target_spans > 0

        for fold in folds:
            if fold["group_count"] == 0:
                empty_fold_penalty += 1e6

            group_cost += ((fold["group_count"] - target_groups) / max(1.0, target_groups)) ** 2
            example_cost += ((fold["example_count"] - target_examples) / max(1.0, target_examples)) ** 2

            if num_labels:
                if np.any(valid_presence):
                    presence_cost += float(
                        np.mean(
                            (
                                (fold["presence"][valid_presence] - target_presence[valid_presence])
                                / target_presence[valid_presence]
                            )
                            ** 2
                        )
                    )
                    missing_label_penalty += float(np.sum(fold["presence"][valid_presence] == 0.0))
                if np.any(valid_spans):
                    span_cost += float(
                        np.mean(
                            (
                                (fold["span_counts"][valid_spans] - target_spans[valid_spans])
                                / target_spans[valid_spans]
                            )
                            ** 2
                        )
                    )

        total_cost = (
            group_cost
            + example_cost
            + (3.0 * presence_cost)
            + (2.0 * span_cost)
            + (10.0 * missing_label_penalty)
            + empty_fold_penalty
        )
        return total_cost

    def _fold_candidate_key(self, fold, profile, target_groups, target_examples, target_presence, target_spans):
        projected_fold = {
            "group_count": fold["group_count"] + 1,
            "example_count": fold["example_count"] + profile["num_examples"],
            "presence": fold["presence"] + profile["presence"],
            "span_counts": fold["span_counts"] + profile["span_counts"],
        }

        group_cost = ((projected_fold["group_count"] - target_groups) / max(1.0, target_groups)) ** 2
        example_cost = ((projected_fold["example_count"] - target_examples) / max(1.0, target_examples)) ** 2

        if len(target_presence):
            valid_presence = target_presence > 0
            if np.any(valid_presence):
                presence_cost = float(
                    np.mean(
                        (
                            (projected_fold["presence"][valid_presence] - target_presence[valid_presence])
                            / target_presence[valid_presence]
                        )
                        ** 2
                    )
                )
                missing_label_penalty = float(np.sum(projected_fold["presence"][valid_presence] == 0.0))
            else:
                presence_cost = 0.0
                missing_label_penalty = 0.0

            valid_spans = target_spans > 0
            if np.any(valid_spans):
                span_cost = float(
                    np.mean(
                        (
                            (projected_fold["span_counts"][valid_spans] - target_spans[valid_spans])
                            / target_spans[valid_spans]
                        )
                        ** 2
                    )
                )
            else:
                span_cost = 0.0
        else:
            presence_cost = 0.0
            span_cost = 0.0
            missing_label_penalty = 0.0

        total_cost = group_cost + example_cost + (3.0 * presence_cost) + (2.0 * span_cost) + (10.0 * missing_label_penalty)
        return (total_cost, projected_fold["group_count"], projected_fold["example_count"])

    def _refine_folds(
        self,
        folds,
        group_profiles,
        target_groups,
        target_examples,
        target_presence,
        target_spans,
        max_groups_per_fold,
    ):
        profile_by_group = {profile["group_id"]: profile for profile in group_profiles}
        best_cost = self._cost_components(folds, target_groups, target_examples, target_presence, target_spans)
        max_passes = 8

        for _ in range(max_passes):
            improved = False

            for source_idx, source_fold in enumerate(folds):
                source_group_ids = list(source_fold["group_ids"])
                for group_id in source_group_ids:
                    if source_fold["group_count"] <= 1:
                        continue

                    profile = profile_by_group[group_id]
                    for target_idx, target_fold in enumerate(folds):
                        if target_idx == source_idx:
                            continue
                        if target_fold["group_count"] >= max_groups_per_fold[target_idx]:
                            continue

                        self._remove_profile(source_fold, profile)
                        self._add_profile(target_fold, profile)
                        candidate_cost = self._cost_components(
                            folds,
                            target_groups,
                            target_examples,
                            target_presence,
                            target_spans,
                        )

                        if candidate_cost + 1e-12 < best_cost:
                            best_cost = candidate_cost
                            improved = True
                            break

                        self._remove_profile(target_fold, profile)
                        self._add_profile(source_fold, profile)

                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            for left_idx in range(len(folds)):
                left_group_ids = list(folds[left_idx]["group_ids"])
                for right_idx in range(left_idx + 1, len(folds)):
                    right_group_ids = list(folds[right_idx]["group_ids"])
                    for left_group_id in left_group_ids:
                        left_profile = profile_by_group[left_group_id]
                        for right_group_id in right_group_ids:
                            right_profile = profile_by_group[right_group_id]

                            self._remove_profile(folds[left_idx], left_profile)
                            self._remove_profile(folds[right_idx], right_profile)
                            self._add_profile(folds[left_idx], right_profile)
                            self._add_profile(folds[right_idx], left_profile)

                            candidate_cost = self._cost_components(
                                folds,
                                target_groups,
                                target_examples,
                                target_presence,
                                target_spans,
                            )

                            if candidate_cost + 1e-12 < best_cost:
                                best_cost = candidate_cost
                                improved = True
                                break

                            self._remove_profile(folds[left_idx], right_profile)
                            self._remove_profile(folds[right_idx], left_profile)
                            self._add_profile(folds[left_idx], left_profile)
                            self._add_profile(folds[right_idx], right_profile)

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

    def split(self, dataset, groups=None):
        if groups is None:
            raise ValueError("groups must be provided for StratifiedGroupKFoldNER")

        groups = np.asarray(groups)
        group_to_indices = defaultdict(list)
        for index, group in enumerate(groups.tolist()):
            group_to_indices[group].append(index)

        if len(group_to_indices) < self.n_splits:
            raise ValueError(
                f"Need at least {self.n_splits} unique groups, found {len(group_to_indices)}."
            )

        label_names = sorted({label for sample in dataset for _, _, label in sample.get("ner", [])})
        label_to_idx = {label: idx for idx, label in enumerate(label_names)}
        num_labels = len(label_names)

        group_profiles = [
            self._build_profile(group_id, indices, dataset, label_to_idx, num_labels)
            for group_id, indices in group_to_indices.items()
        ]

        global_presence = np.zeros(num_labels, dtype=float)
        global_spans = np.zeros(num_labels, dtype=float)
        for profile in group_profiles:
            global_presence += profile["presence"]
            global_spans += profile["span_counts"]

        rarity_scores = []
        for profile in group_profiles:
            rarity = 0.0
            for label_idx, present in enumerate(profile["presence"]):
                if present:
                    rarity += 1.0 / max(1.0, global_presence[label_idx])
            rarity_scores.append(rarity)

        rng = np.random.default_rng(self.seed)
        shuffled_order = rng.permutation(len(group_profiles)).tolist()
        order = sorted(
            shuffled_order,
            key=lambda idx: (
                rarity_scores[idx],
                group_profiles[idx]["num_spans"],
                group_profiles[idx]["num_examples"],
            ),
            reverse=True,
        )

        target_groups = len(group_profiles) / self.n_splits
        target_examples = sum(profile["num_examples"] for profile in group_profiles) / self.n_splits
        target_presence = global_presence / self.n_splits if num_labels else np.asarray([])
        target_spans = global_spans / self.n_splits if num_labels else np.asarray([])
        max_groups_per_fold = self._group_capacity_limits(len(group_profiles))

        folds = [self._empty_fold(num_labels) for _ in range(self.n_splits)]

        for seed_fold_idx, profile_idx in enumerate(order[: self.n_splits]):
            self._add_profile(folds[seed_fold_idx], group_profiles[profile_idx])

        for profile_idx in order[self.n_splits :]:
            profile = group_profiles[profile_idx]
            candidate_folds = [
                fold_idx
                for fold_idx in range(self.n_splits)
                if folds[fold_idx]["group_count"] < max_groups_per_fold[fold_idx]
            ]
            best_fold_idx = min(
                candidate_folds,
                key=lambda fold_idx: (
                    *self._fold_candidate_key(
                        folds[fold_idx],
                        profile,
                        target_groups,
                        target_examples,
                        target_presence,
                        target_spans,
                    ),
                    fold_idx,
                ),
            )
            self._add_profile(folds[best_fold_idx], profile)

        self._refine_folds(
            folds,
            group_profiles,
            target_groups,
            target_examples,
            target_presence,
            target_spans,
            max_groups_per_fold,
        )

        if any(fold["group_count"] == 0 for fold in folds):
            raise RuntimeError("Internal error: stratified group splitter produced an empty fold.")

        self.last_summary = {
            "label_names": label_names,
            "folds": [
                {
                    "fold": fold_idx + 1,
                    "group_count": fold["group_count"],
                    "example_count": fold["example_count"],
                    "label_presence": {
                        label: int(fold["presence"][label_idx])
                        for label_idx, label in enumerate(label_names)
                    },
                    "label_spans": {
                        label: int(fold["span_counts"][label_idx])
                        for label_idx, label in enumerate(label_names)
                    },
                }
                for fold_idx, fold in enumerate(folds)
            ],
        }

        for fold in folds:
            test_indices = np.asarray(sorted(fold["indices"]), dtype=int)
            train_indices = np.asarray(
                sorted(index for other in folds if other is not fold for index in other["indices"]),
                dtype=int,
            )
            yield train_indices, test_indices
