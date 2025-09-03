DETECTOR_REGISTRY: dict[str, type] = {}

def register_detector(name: str):
    def decorator(cls: type) -> type:
        DETECTOR_REGISTRY[name] = cls
        return cls
    return decorator
