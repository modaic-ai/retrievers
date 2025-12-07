class SchemaError(Exception):
    """Raised when a schema is invalid"""

    pass


class BackendCompatibilityError(Exception):
    """Raised when a feature is not supported by a backend"""

    pass
