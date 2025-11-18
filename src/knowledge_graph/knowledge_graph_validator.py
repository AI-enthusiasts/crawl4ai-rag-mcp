"""
Knowledge Graph Validator

Validates AI-generated code against Neo4j knowledge graph containing
repository information. Checks imports, methods, attributes, and parameters.
"""

import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

try:
    from neo4j import NotificationMinimumSeverity
except ImportError:
    NotificationMinimumSeverity = None  # type: ignore[assignment]

from .ai_script_analyzer import AnalysisResult
from .validation import (
    ScriptValidationResult,
    ValidationResult,
    ValidationStatus,
    neo4j_queries,
)
from .validation.attribute_validator import (
    AttributeValidation,
    validate_attribute_accesses,
    validate_single_attribute_access,
)
from .validation.class_validator import (
    ClassValidation,
    validate_class_instantiations,
    validate_single_class_instantiation,
)
from .validation.function_validator import (
    FunctionValidation,
    validate_function_calls,
    validate_single_function_call,
)
from .validation.import_validator import (
    ImportValidation,
    validate_imports,
    validate_single_import,
)
from .validation.method_validator import (
    MethodValidation,
    validate_method_calls,
    validate_single_method_call,
)
from .validation.utils import (
    calculate_overall_confidence,
    detect_hallucinations,
    is_from_knowledge_graph,
    validate_parameters,
)

logger = logging.getLogger(__name__)


class KnowledgeGraphValidator:
    """Validates code against Neo4j knowledge graph"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver: AsyncDriver | None = None

        # Cache for performance
        self.module_cache: dict[str, list[str]] = {}
        self.class_cache: dict[str, dict[str, Any]] = {}
        self.method_cache: dict[str, list[dict[str, Any]]] = {}
        self.repo_cache: dict[str, str | None] = {}  # module_name -> repo_name
        self.knowledge_graph_modules: set[str] = set()  # Modules in kg

    async def initialize(self) -> None:
        """Initialize Neo4j connection"""
        if NotificationMinimumSeverity is not None:
            try:
                self.driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                    warn_notification_severity=NotificationMinimumSeverity.OFF,
                )
            except (ImportError, AttributeError):
                logging.getLogger("neo4j.notifications").setLevel(
                    logging.ERROR,
                )
                self.driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                )
        else:
            logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
            self.driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
        logger.info("Knowledge graph validator initialized")

    async def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()

    async def validate_script(
        self, analysis_result: AnalysisResult,
    ) -> ScriptValidationResult:
        """Validate entire script analysis against knowledge graph"""
        result = ScriptValidationResult(
            script_path=analysis_result.file_path,
            analysis_result=analysis_result,
        )

        # Validate imports first (builds context for other validations)
        result.import_validations = await self._validate_imports(
            analysis_result.imports,
        )

        # Validate class instantiations
        result.class_validations = await self._validate_class_instantiations(
            analysis_result.class_instantiations,
        )

        # Validate method calls
        result.method_validations = await self._validate_method_calls(
            analysis_result.method_calls,
        )

        # Validate attribute accesses
        result.attribute_validations = await self._validate_attribute_accesses(
            analysis_result.attribute_accesses,
        )

        # Validate function calls
        result.function_validations = await self._validate_function_calls(
            analysis_result.function_calls,
        )

        # Calculate overall confidence and detect hallucinations
        result.overall_confidence = self._calculate_overall_confidence(result)
        result.hallucinations_detected = self._detect_hallucinations(result)

        return result

    # Delegation methods - Import validation

    async def _validate_imports(self, imports: Any) -> list[ImportValidation]:
        """Delegate to validation.import_validator.validate_imports"""
        return await validate_imports(
            imports,
            lambda m: neo4j_queries.find_modules(self.driver, m),
            lambda m: neo4j_queries.get_module_contents(self.driver, m),
            self.module_cache,
            self.knowledge_graph_modules,
        )

    async def _validate_single_import(self, import_info: Any) -> ImportValidation:
        """Delegate to validation.import_validator.validate_single_import"""
        return await validate_single_import(
            import_info,
            lambda m: neo4j_queries.find_modules(self.driver, m),
            lambda m: neo4j_queries.get_module_contents(self.driver, m),
            self.module_cache,
            self.knowledge_graph_modules,
        )

    # Delegation methods - Class validation

    async def _validate_class_instantiations(
        self, instantiations: Any,
    ) -> list[ClassValidation]:
        """Delegate to validation.class_validator.validate_class_instantiations"""
        return await validate_class_instantiations(
            instantiations,
            lambda cn: neo4j_queries.find_class(
                self.driver, cn, self.repo_cache,
            ),
            lambda cn, mn: neo4j_queries.find_method(
                self.driver, cn, mn, self.method_cache, self.repo_cache,
            ),
            validate_parameters,
            lambda ct: is_from_knowledge_graph(
                ct, self.knowledge_graph_modules,
            ),
            self.knowledge_graph_modules,
        )

    async def _validate_single_class_instantiation(
        self, instantiation: Any,
    ) -> ClassValidation:
        """Validate single class instantiation against knowledge graph"""
        return await validate_single_class_instantiation(
            instantiation,
            lambda cn: neo4j_queries.find_class(
                self.driver, cn, self.repo_cache,
            ),
            lambda cn, mn: neo4j_queries.find_method(
                self.driver, cn, mn, self.method_cache, self.repo_cache,
            ),
            validate_parameters,
            lambda ct: is_from_knowledge_graph(
                ct, self.knowledge_graph_modules,
            ),
            self.knowledge_graph_modules,
        )

    # Delegation methods - Method validation

    async def _validate_method_calls(
        self, method_calls: Any,
    ) -> list[MethodValidation]:
        """Validate method calls against knowledge graph"""
        return await validate_method_calls(
            method_calls,
            lambda cn, mn: neo4j_queries.find_method(
                self.driver, cn, mn, self.method_cache, self.repo_cache,
            ),
            lambda cn, mn: neo4j_queries.find_similar_methods(
                self.driver, cn, mn, self.repo_cache,
            ),
            validate_parameters,
            lambda ct: is_from_knowledge_graph(
                ct, self.knowledge_graph_modules,
            ),
            self.knowledge_graph_modules,
        )

    async def _validate_single_method_call(
        self, method_call: Any,
    ) -> MethodValidation:
        """Validate single method call against knowledge graph"""
        return await validate_single_method_call(
            method_call,
            lambda cn, mn: neo4j_queries.find_method(
                self.driver, cn, mn, self.method_cache, self.repo_cache,
            ),
            lambda cn, mn: neo4j_queries.find_similar_methods(
                self.driver, cn, mn, self.repo_cache,
            ),
            validate_parameters,
            lambda ct: is_from_knowledge_graph(
                ct, self.knowledge_graph_modules,
            ),
            self.knowledge_graph_modules,
        )

    # Delegation methods - Attribute validation

    async def _validate_attribute_accesses(
        self, attribute_accesses: Any,
    ) -> list[AttributeValidation]:
        """Validate attribute accesses against knowledge graph"""
        return await validate_attribute_accesses(
            attribute_accesses,
            lambda cn, an: neo4j_queries.find_attribute(
                self.driver, cn, an, self.repo_cache,
            ),
            lambda cn, mn: neo4j_queries.find_method(
                self.driver, cn, mn, self.method_cache, self.repo_cache,
            ),
            lambda ct: is_from_knowledge_graph(
                ct, self.knowledge_graph_modules,
            ),
            self.knowledge_graph_modules,
        )

    async def _validate_single_attribute_access(
        self, attr_access: Any,
    ) -> AttributeValidation:
        """Validate single attribute access against knowledge graph"""
        return await validate_single_attribute_access(
            attr_access,
            lambda cn, an: neo4j_queries.find_attribute(
                self.driver, cn, an, self.repo_cache,
            ),
            lambda cn, mn: neo4j_queries.find_method(
                self.driver, cn, mn, self.method_cache, self.repo_cache,
            ),
            lambda ct: is_from_knowledge_graph(
                ct, self.knowledge_graph_modules,
            ),
            self.knowledge_graph_modules,
        )

    # Delegation methods - Function validation

    async def _validate_function_calls(
        self, function_calls: Any,
    ) -> list[FunctionValidation]:
        """Validate function calls against knowledge graph"""
        return await validate_function_calls(
            function_calls,
            lambda fn: neo4j_queries.find_function(
                self.driver, fn, self.repo_cache,
            ),
            validate_parameters,
            lambda fn: is_from_knowledge_graph(
                fn, self.knowledge_graph_modules,
            ),
            self.knowledge_graph_modules,
        )

    async def _validate_single_function_call(
        self, func_call: Any,
    ) -> FunctionValidation:
        """Validate single function call against knowledge graph"""
        return await validate_single_function_call(
            func_call,
            lambda fn: neo4j_queries.find_function(
                self.driver, fn, self.repo_cache,
            ),
            validate_parameters,
            lambda fn: is_from_knowledge_graph(
                fn, self.knowledge_graph_modules,
            ),
            self.knowledge_graph_modules,
        )

    # Delegation methods - Utility functions

    def _calculate_overall_confidence(self, result: ScriptValidationResult) -> float:
        """Delegate to validation.utils.calculate_overall_confidence"""
        return calculate_overall_confidence(
            result,
            self.knowledge_graph_modules,
        )

    def _is_from_knowledge_graph(self, class_type: str) -> bool:
        """Delegate to validation.utils.is_from_knowledge_graph"""
        return is_from_knowledge_graph(class_type, self.knowledge_graph_modules)

    def _detect_hallucinations(
        self,
        result: ScriptValidationResult,
    ) -> list[dict[str, Any]]:
        """Delegate to validation.utils.detect_hallucinations"""
        return detect_hallucinations(
            result,
            self.knowledge_graph_modules,
        )


__all__ = [
    "KnowledgeGraphValidator",
    "ScriptValidationResult",
    "ValidationResult",
    "ValidationStatus",
]
