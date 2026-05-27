PUBLIC_MODEL_CHOICES = ["esm1v", "esm2", "esmif", "vaspa", "poet", "prosst"]
STRUCTURE_TOKEN_MODEL = "prosst"


def normalize_model_name(model_name):
    normalized = str(model_name).strip().lower()
    if normalized not in PUBLIC_MODEL_CHOICES:
        choices = ", ".join(PUBLIC_MODEL_CHOICES)
        raise ValueError(f"Unknown model '{model_name}'. Use one of: {choices}.")
    return normalized


def normalize_model_names(model_names):
    return [normalize_model_name(model_name) for model_name in model_names or []]
