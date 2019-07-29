from importlib import import_module



def factory(architecture_name, *args, **kwargs):
    architecture_module = import_module(
        '.architectures.' + architecture_name, package='low_shot_learning')
    create_model = getattr(architecture_module, 'create_model')
    return create_model(*args, **kwargs)
