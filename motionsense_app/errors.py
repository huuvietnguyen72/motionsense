from typing import Any


class DomainError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status: int = 422,
        details: list | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or []


def api_error(code: str, message: str, details: list[Any] | None = None) -> dict:
    normalized = []
    for detail in details or []:
        if isinstance(detail, dict):
            normalized.append(
                {
                    "row": detail.get("row"),
                    "column": detail.get("column"),
                    "message": detail.get("message", "Dữ liệu không hợp lệ."),
                }
            )
        else:
            normalized.append({"row": None, "column": None, "message": str(detail)})
    return {"error": {"code": code, "message": message, "details": normalized}}
