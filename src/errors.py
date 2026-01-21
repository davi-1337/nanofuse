class NanoFuseError(Exception):
    pass


class ModelFileNotFoundError(NanoFuseError):
    pass


class ModelDimensionMismatchError(NanoFuseError):
    pass


class TensorShapeMismatchError(NanoFuseError):
    pass


class QuantizationError(NanoFuseError):
    pass


class TokenizerError(NanoFuseError):
    pass
