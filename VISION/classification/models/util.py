import importlib

def load_model(**kwargs):
    num_classes = kwargs["num_classes"]
    model_name = kwargs["model_name"]
    pretrained = kwargs["pretrained"]

    if "efficientnet" in model_name.split("-"):
        module = importlib.import_module(f"models.efficientnet")
        get_model_func = getattr(module, "get_model")
        model = get_model_func(model_name=model_name, pretrained=pretrained, num_classes=num_classes)

    else:
        module = importlib.import_module(f"models.{model_name}")
        get_model_func = getattr(module, "get_model")
        model = get_model_func(num_classes=num_classes)

    return model