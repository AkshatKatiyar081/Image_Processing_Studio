class PipelineError(Exception):
    """Raised for invalid input or parameters. `code` is a stable machine-readable string."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}
