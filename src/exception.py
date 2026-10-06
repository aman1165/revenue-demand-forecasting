class ForecastingException(Exception):
    """Project-specific exception for clearer error reporting."""

    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self) -> str:
        return f"ForecastingException: {self.args[0]}"
