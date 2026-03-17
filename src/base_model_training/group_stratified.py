from collections import defaultdict

import numpy as np


class StratifiedGroupKFoldNER:
    """Heuristic group-aware multilabel splitter for NER datasets."""

    def __init__(self, n_splits, seed=42):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.seed = seed
        self.last_summary = None

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
        group_ids = list(group_to_indices.keys())

        group_profiles = []
        global_presence = np.zeros(len(label_names), dtype=float)
        global_spans = np.zeros(len(label_names), dtype=float)

        for group_id in group_ids:
            indices = group_to_indices[group_id]
            presence = np.zeros(len(label_names), dtype=float)
            span_counts = np.zeros(len(label_names), dtype=float)
            for index in indices:
                for _start, _end, label in dataset[index].get("ner", []):
                    label_idx = label_to_idx[label]
                    presence[label_idx] = 1.0
                    span_counts[label_idx] += 1.0
            global_presence += presence
            global_spans += span_counts
            group_profiles.append(
                {
                    "group_id": group_id,
                    "indices": indices,
                    "presence": presence,
                    "span_counts": span_counts,
                    "num_examples": len(indices),
                    "num_spans": int(span_counts.sum()),
                }
            )

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(group_profiles)).tolist()
        rarity_scores = []
        for profile in group_profiles:
            rarity = 0.0
            for label_idx, present in enumerate(profile["presence"]):
                if present:
                    rarity += 1.0 / max(1.0, global_presence[label_idx])
            rarity_scores.append(rarity)

        order.sort(
            key=lambda idx: (
                rarity_scores[idx],
                group_profiles[idx]["num_spans"],
                group_profiles[idx]["num_examples"],
            ),
            reverse=True,
        )

        target_groups = len(group_profiles) / self.n_splits
        target_examples = sum(profile["num_examples"] for profile in group_profiles) / self.n_splits
        target_presence = global_presence / self.n_splits if len(label_names) else np.asarray([])
        target_spans = global_spans / self.n_splits if len(label_names) else np.asarray([])

        folds = []
        for _ in range(self.n_splits):
            folds.append(
                {
                    "group_ids": [],
                    "indices": [],
                    "presence": np.zeros(len(label_names), dtype=float),
                    "span_counts": np.zeros(len(label_names), dtype=float),
                    "group_count": 0,
                    "example_count": 0,
                }
            )

        for profile_idx in order:
            profile = group_profiles[profile_idx]
            best_fold_idx = None
            best_key = None

            for fold_idx, fold in enumerate(folds):
                projected_presence = fold["presence"] + profile["presence"]
                projected_spans = fold["span_counts"] + profile["span_counts"]
                projected_group_count = fold["group_count"] + 1
                projected_example_count = fold["example_count"] + profile["num_examples"]

                group_cost = ((projected_group_count - target_groups) / max(1.0, target_groups)) ** 2
                example_cost = ((projected_example_count - target_examples) / max(1.0, target_examples)) ** 2

                if len(label_names):
                    valid_presence = target_presence > 0
                    if np.any(valid_presence):
                        presence_cost = float(
                            np.mean(
                                (
                                    (projected_presence[valid_presence] - target_presence[valid_presence])
                                    / target_presence[valid_presence]
                                )
                                ** 2
                            )
                        )
                    else:
                        presence_cost = 0.0

                    valid_spans = target_spans > 0
                    if np.any(valid_spans):
                        span_cost = float(
                            np.mean(
                                (
                                    (projected_spans[valid_spans] - target_spans[valid_spans])
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

                total_cost = group_cost + example_cost + (3.0 * presence_cost) + (2.0 * span_cost)
                candidate_key = (
                    total_cost,
                    projected_group_count,
                    projected_example_count,
                    fold_idx,
                )
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_fold_idx = fold_idx

            chosen_fold = folds[best_fold_idx]
            chosen_fold["group_ids"].append(profile["group_id"])
            chosen_fold["indices"].extend(profile["indices"])
            chosen_fold["presence"] += profile["presence"]
            chosen_fold["span_counts"] += profile["span_counts"]
            chosen_fold["group_count"] += 1
            chosen_fold["example_count"] += profile["num_examples"]

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
