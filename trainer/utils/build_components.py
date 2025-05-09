import importlib

def get_class(path, name):
    module = importlib.import_module(path)
    return getattr(module, name)

def build_module(config):

    if config is None:
        return None
    
    """Build a module from configuration."""
    class_ = get_class(config['path'], config['name'])
    
    if 'pretrained' in config and hasattr(class_, 'from_pretrained'):
        return class_.from_pretrained(config['pretrained'])
    elif 'config' in config and hasattr(class_, 'from_config'):
        return class_.from_config(config['config'])
    elif 'params' in config:
        return class_(**config['params'])
    else:
        raise ValueError("Configuration must contain 'pretrained', 'config', or 'params' key.")

def build_modules(configs):
    """Build multiple modules from configuration."""
    modules = {}
    for config in configs:
        modules[config['name']] = build_module(config)
    return modules

def build_optimizer(config, model_parameters):
    """Build a module from configuration."""
    class_ = get_class(config['path'], config['name'])
    
    if 'params' in config:
        return class_(model_parameters, **config['params'])
    else:
        raise ValueError("Configuration must contain 'pretrained', 'config', or 'params' key.")
