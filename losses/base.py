class BaseLoss:
    """Base class for all losses, this is to ensure spesific mechanisms that should apply to all."""

    def __init_subclass__(cls, **kwargs):
        """Ensure all subclasses implement get_alias method"""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "get_alias") or not callable(getattr(cls, "get_alias")):
            raise NotImplementedError(f"{cls.__name__} must implement get_alias method")
        if not hasattr(cls, "__call__") or not callable(getattr(cls, "__call__")):
            raise NotImplementedError(f"{cls.__name__} must implement __call__ method")
