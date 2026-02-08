import itertools


def generate_trial_params(config, rng):
    """Generate (lr, weight_decay) pairs according to configured search mode."""
    candidates = list(itertools.product(config.lr_values, config.weight_decay_values))
    if not candidates:
        raise ValueError("No hyperparameter candidates were generated.")

    if config.search_mode == "grid":
        return candidates

    if config.search_mode == "random":
        count = min(config.num_trials, len(candidates))
        sampled_indices = rng.choice(len(candidates), size=count, replace=False)
        return [candidates[index] for index in sampled_indices]

    raise ValueError(f"Unsupported search_mode: {config.search_mode}")
