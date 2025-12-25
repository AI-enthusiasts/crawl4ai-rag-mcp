from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class ErrorType(Enum):
    SYNTAX = 'syntax'
    SEMANTIC = 'semantic'
    RUNTIME = 'runtime'

class Severity(Enum):
    ERROR = 'error'
    WARNING = 'warning'
    INFO = 'info'

@dataclass
class Suggestion:
    message: str
    fix: str | None = ...
    def to_dict(self) -> dict: ...

@dataclass
class ErrorDetail:
    type: ErrorType
    code: str
    severity: Severity
    message: str
    line: int
    column: int
    source_line: str
    end_line: int | None = ...
    end_column: int | None = ...
    line_before: str | None = ...
    line_after: str | None = ...
    suggestions: list[Suggestion] = field(default_factory=list)
    documentation_url: str | None = ...
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
    @property
    def formatted_message(self) -> str: ...
    @property
    def simple_message(self) -> str: ...

@dataclass
class WarningDetail:
    code: str
    message: str
    line: int
    column: int
    def to_dict(self) -> dict: ...

@dataclass
class CompilationResult:
    success: bool
    js_code: list[str] | None = ...
    errors: list[ErrorDetail] = field(default_factory=list)
    warnings: list[WarningDetail] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
    @property
    def has_errors(self) -> bool: ...
    @property
    def has_warnings(self) -> bool: ...
    @property
    def first_error(self) -> ErrorDetail | None: ...

@dataclass
class ValidationResult:
    valid: bool
    errors: list[ErrorDetail] = field(default_factory=list)
    warnings: list[WarningDetail] = field(default_factory=list)
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
    @property
    def first_error(self) -> ErrorDetail | None: ...
