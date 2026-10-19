"""Explicit experiment model routing and response identity checks."""
SUPPORTED_MODELS = ('gpt-6-astra', 'gpt-5.5-2026-04-23', 'gpt-5.6-sol')


def checked_model(model):
    if model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported experiment model: '+str(model))
    return model


def model_matches(expected, returned):
    return isinstance(returned,str) and (returned==expected or returned.startswith(expected+'-'))


def bind_experiment_profile(config):
    """Bind the factory, compactor and vision resolver only in this worker process."""
    import qwenpaw.config.config as configuration
    from qwenpaw.app.agent_context import set_current_agent_id
    from qwenpaw.providers import ProviderManager
    from qwenpaw.providers.provider import ModelInfo
    checked_model(config.active_model.model)
    original=configuration.load_agent_config
    def resolve(agent_id):
        return config if agent_id==config.id else original(agent_id)
    configuration.load_agent_config=resolve
    set_current_agent_id(config.id)
    provider=ProviderManager.get_instance().get_provider(config.active_model.provider_id)
    if provider is None:
        raise RuntimeError('Pinned experiment provider is unavailable')
    selected=next((m for m in provider.models+provider.extra_models if m.id==config.active_model.model),None)
    if selected is None:
        selected=ModelInfo(id=config.active_model.model,name=config.active_model.model,
            supports_multimodal=True,supports_image=True,supports_video=False,probe_source='probed',
            generate_kwargs={'api_mode':'responses'})
        provider.extra_models.append(selected)
    else:
        selected.supports_multimodal=True
        selected.supports_image=True
        selected.generate_kwargs={**selected.generate_kwargs,'api_mode':'responses'}
    return original
