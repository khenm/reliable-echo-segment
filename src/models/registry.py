import torch.nn as nn

class ModelRegistry:
    """
    Central registry for model classes.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        """
        Decorator to register a model class with a specific name.
        """
        def decorator(model_class):
            upper_name = name.upper()
            if upper_name in cls._registry:
                raise ValueError(f"Model '{upper_name}' is already registered.")
            cls._registry[upper_name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name):
        """
        Retrieve a model class by name.
        """
        upper_name = name.upper()
        if upper_name not in cls._registry:
            raise ValueError(f"Model '{upper_name}' is not registered. Available models: {list(cls._registry.keys())}")
        return cls._registry[upper_name]

    @classmethod
    def build(cls, cfg, device):
        """
        Instantiate a model based on the configuration.

        Args:
            cfg (dict): Configuration dictionary.
            device (torch.device): Device to load the model onto.

        Returns:
            nn.Module: Instantiated model.
        """
        model_name = cfg['model'].get('name', 'VAEUNet') # Default fallback
        model_class = cls.get(model_name)
        
        if hasattr(model_class, 'from_config'):
            return model_class.from_config(cfg).to(device)
        else:
            raise NotImplementedError(f"Model '{model_name}' does not implement 'from_config'.")


def get_model(cfg, device):
    """
    Instantiates and returns the model based on configuration using ModelRegistry.
    
    Args:
        cfg (dict): Configuration dictionary defining model hyperparameters.
        device (torch.device): Device to load the model onto.
        
    Returns:
        torch.nn.Module: The instantiated model.
    """
    return ModelRegistry.build(cfg, device)