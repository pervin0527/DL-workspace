import importlib

def load_model(**kwargs):
    num_classes = kwargs["num_classes"]
    model_name = kwargs["model_name"]

    module = importlib.import_module(f"models.{model_name}")
    get_model_func = getattr(module, "get_model")

    model = get_model_func(num_classes=num_classes)

    return model