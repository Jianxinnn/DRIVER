import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Keep normalization importable in lightweight environments.
    torch = None


def check_torch2numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def select_closest_values(results, esm2_results, vaspa_results, select_num=9):
    selected_results = []
    selected_esm2_results = []
    selected_vaspa_results = []

    for result, esm2_result, vaspa_result in zip(results, esm2_results, vaspa_results):
        diffs = [(abs(val - result[0]), idx) for idx, val in enumerate(result)]
        diffs.sort()
        selected_indices = [idx for _, idx in diffs[:select_num]]
        selected_result = [result[idx] for idx in selected_indices]
        selected_esm2_result = [esm2_result[idx] for idx in selected_indices]
        selected_vaspa_result = [vaspa_result[idx] for idx in selected_indices]
        selected_results.append(selected_result)
        selected_esm2_results.append(selected_esm2_result)
        selected_vaspa_results.append(selected_vaspa_result)
    return selected_results, selected_esm2_results, selected_vaspa_results

def select_for_covery(results, select_num=9):
    select_results = []
    for result in results:
        select_results.append(result[:select_num])
    return select_results


def _resolve_normalization_method(method):
    aliases = {
        "minmax": "minmax",
        "min-max": "minmax",
        "0-1": "minmax",
        "01": "minmax",
        "zeroone": "minmax",
        "zero-one": "minmax",
        "zscore": "zscore",
        "z-score": "zscore",
        "standard": "zscore",
        "standardize": "zscore",
    }
    key = str(method or "0-1").strip().lower().replace("_", "-")
    if key not in aliases:
        raise ValueError(
            f"Unknown normalization method: {method}. Use 'minmax' or 'zscore'."
        )
    return aliases[key]


def normalize_data_sets(data_sets, method="0-1"):
    """Apply per-model score normalization in place.

    The function name is kept for backwards compatibility with existing callers.
    Supported methods:
      - 0-1/minmax: range normalization, default
      - zscore: mean/std standardization
    """
    method = _resolve_normalization_method(method)
    for data in data_sets:
        if len(data) == 0:
            continue
        first_elements = np.asarray(
            [np.asarray(subset[0], dtype=float).reshape(-1)[0] for subset in data],
            dtype=float,
        )
        if method == "minmax":
            center = float(np.min(first_elements))
            scale = float(np.max(first_elements) - center)
        else:
            center = float(np.mean(first_elements))
            scale = float(np.std(first_elements))

        if scale == 0:
            for i, subset in enumerate(data):
                data[i] = [np.asarray(x, dtype=float) * 0 for x in subset]
            continue

        for i, subset in enumerate(data):
            data[i] = [
                (np.asarray(x, dtype=float) - center) / scale
                for x in subset
            ]
            
def handle_empty_lists(results_list, data_shape):
    if not results_list:
        return [None] * data_shape
    else:
        return results_list
    

if __name__ == '__main__':
    exit(0)
