"""
Comprehensive tests for KnowledgeGraphValidator functionality.

This module tests knowledge graph validation including:
- Script validation against Neo4j knowledge graph
- Import validation and hallucination detection
- Method call validation with parameter checking
- Class instantiation validation
- Function call validation
- Attribute access validation
- Confidence scoring and reporting
- Caching mechanisms
- Error handling and edge cases
"""

import os

# Add src to path for imports
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import fixtures
from tests.fixtures.neo4j_fixtures import (
    MockNeo4jDriver,
)

# Mock Neo4j and other dependencies before importing our modules
with patch.dict(
    "sys.modules",
    {
        "neo4j": MagicMock(),
        "neo4j.AsyncGraphDatabase": MagicMock(),
        "ai_script_analyzer": MagicMock(),
        "hallucination_reporter": MagicMock(),
    },
):
    # Now import our modules
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(__file__), "..", "knowledge_graphs"),
    )
    from src.knowledge_graph.knowledge_graph_validator import (
        ImportValidation,
        KnowledgeGraphValidator,
        ValidationResult,
        ValidationStatus,
    )


class TestValidationDataClasses:
    """Test the validation data classes and enums"""

    def test_validation_status_enum(self):
        """Test ValidationStatus enum values"""
        assert ValidationStatus.VALID.value == "VALID"
        assert ValidationStatus.INVALID.value == "INVALID"
        assert ValidationStatus.UNCERTAIN.value == "UNCERTAIN"
        assert ValidationStatus.NOT_FOUND.value == "NOT_FOUND"

    def test_validation_result_creation(self):
        """Test ValidationResult dataclass creation"""
        result = ValidationResult(
            status=ValidationStatus.VALID,
            confidence=0.9,
            message="Test validation result",
            details={"key": "value"},
            suggestions=["suggestion1", "suggestion2"],
        )

        assert result.status == ValidationStatus.VALID
        assert result.confidence == 0.9
        assert result.message == "Test validation result"
        assert result.details == {"key": "value"}
        assert result.suggestions == ["suggestion1", "suggestion2"]

    def test_validation_result_defaults(self):
        """Test ValidationResult with default values"""
        result = ValidationResult(
            status=ValidationStatus.VALID,
            confidence=1.0,
            message="Test message",
        )

        assert result.details == {}
        assert result.suggestions == []

    def test_import_validation_creation(self):
        """Test ImportValidation dataclass creation"""
        # Mock import info
        mock_import_info = MagicMock()
        mock_import_info.module = "test_module"
        mock_import_info.imports = ["TestClass"]

        validation_result = ValidationResult(
            status=ValidationStatus.VALID,
            confidence=0.8,
            message="Import validated",
        )

        import_validation = ImportValidation(
            import_info=mock_import_info,
            validation=validation_result,
            available_classes=["TestClass", "OtherClass"],
            available_functions=["test_function"],
        )

        assert import_validation.import_info == mock_import_info
        assert import_validation.validation == validation_result
        assert "TestClass" in import_validation.available_classes
        assert "test_function" in import_validation.available_functions


@pytest.fixture
def validator_config():
    """Configuration for KnowledgeGraphValidator"""
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "test_user",
        "neo4j_password": "test_password",
    }


class TestKnowledgeGraphValidator:
    """Test the KnowledgeGraphValidator class"""

    @pytest.fixture
    def mock_validator(self, validator_config):
        """Create a KnowledgeGraphValidator with mocked dependencies"""
        with patch("src.knowledge_graph.knowledge_graph_validator.AsyncGraphDatabase") as mock_db:
            mock_driver = MockNeo4jDriver()
            mock_db.driver.return_value = mock_driver

            validator = KnowledgeGraphValidator(
                validator_config["neo4j_uri"],
                validator_config["neo4j_user"],
                validator_config["neo4j_password"],
            )
            validator.driver = mock_driver

            yield validator

    @pytest.mark.asyncio
    async def test_initialization(self, validator_config):
        """Test KnowledgeGraphValidator initialization"""
        with patch("src.knowledge_graph.knowledge_graph_validator.AsyncGraphDatabase") as mock_db:
            mock_driver = MockNeo4jDriver()
            mock_db.driver.return_value = mock_driver

            validator = KnowledgeGraphValidator(
                validator_config["neo4j_uri"],
                validator_config["neo4j_user"],
                validator_config["neo4j_password"],
            )

            assert validator.neo4j_uri == validator_config["neo4j_uri"]
            assert validator.neo4j_user == validator_config["neo4j_user"]
            assert validator.neo4j_password == validator_config["neo4j_password"]
            assert validator.driver is None  # Not initialized yet

            # Check cache initialization
            assert isinstance(validator.module_cache, dict)
            assert isinstance(validator.class_cache, dict)
            assert isinstance(validator.method_cache, dict)
            assert isinstance(validator.repo_cache, dict)
            assert isinstance(validator.knowledge_graph_modules, set)

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_validator):
        """Test successful initialization of validator"""
        await mock_validator.initialize()

        # Verify driver is set up
        assert mock_validator.driver is not None
        assert isinstance(mock_validator.driver, MockNeo4jDriver)

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, validator_config):
        """Test initialization failure due to connection issues"""
        with patch("src.knowledge_graph.knowledge_graph_validator.AsyncGraphDatabase") as mock_db:
            # Simulate connection failure
            mock_db.driver.side_effect = Exception("Connection failed")

            validator = KnowledgeGraphValidator(
                validator_config["neo4j_uri"],
                validator_config["neo4j_user"],
                validator_config["neo4j_password"],
            )

            with pytest.raises(Exception, match="Connection failed"):
                await validator.initialize()

    @pytest.mark.asyncio
    async def test_close_connection(self, mock_validator):
        """Test closing validator connection"""
        await mock_validator.initialize()
        await mock_validator.close()

        assert mock_validator.driver.closed is True

    @pytest.mark.asyncio
    async def test_validate_script_success(self, mock_validator, neo4j_query_responses):
        """Test successful script validation"""
        await mock_validator.initialize()

        # Mock analysis result
        mock_analysis_result = MagicMock()
        mock_analysis_result.imports = []
        mock_analysis_result.class_instantiations = []
        mock_analysis_result.method_calls = []
        mock_analysis_result.attribute_accesses = []
        mock_analysis_result.function_calls = []

        # Set up mock responses
        mock_validator.driver.session_data = neo4j_query_responses["find_module"]

        result = await mock_validator.validate_script(mock_analysis_result)

        assert result is not None
        assert hasattr(result, "overall_confidence")

    @pytest.mark.asyncio
    async def test_validate_imports_valid(self, mock_validator, neo4j_query_responses):
        """Test validation of valid imports"""
        await mock_validator.initialize()

        # Mock import info
        mock_import = MagicMock()
        mock_import.module = "main"
        mock_import.imports = ["TestClass"]

        # Set up mock responses for finding module
        mock_validator.driver.session_data = neo4j_query_responses["find_module"]

        result = await mock_validator._validate_single_import(mock_import)

        assert result is not None
        assert result.validation.status in [
            ValidationStatus.VALID,
            ValidationStatus.NOT_FOUND,
        ]

    @pytest.mark.asyncio
    async def test_validate_imports_invalid(
        self,
        mock_validator,
        neo4j_query_responses,
    ):
        """Test validation of invalid imports"""
        await mock_validator.initialize()

        # Mock invalid import
        mock_import = MagicMock()
        mock_import.module = "nonexistent_module"
        mock_import.imports = ["FakeClass"]

        # Set up empty response (module not found)
        mock_validator.driver.session_data = neo4j_query_responses["empty_result"]

        result = await mock_validator._validate_single_import(mock_import)

        assert result is not None
        assert result.validation.status == ValidationStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_validate_class_instantiation_valid(
        self,
        mock_validator,
        neo4j_query_responses,
    ):
        """Test validation of valid class instantiation"""
        await mock_validator.initialize()

        # Mock class instantiation
        mock_instantiation = MagicMock()
        mock_instantiation.class_name = "TestClass"
        mock_instantiation.module = "main"
        mock_instantiation.args = ["param1"]
        mock_instantiation.kwargs = {}

        # Set up mock responses
        mock_validator.driver.session_data = neo4j_query_responses["find_class"]

        result = await mock_validator._validate_single_class_instantiation(
            mock_instantiation,
        )

        assert result is not None
        assert result.validation.status in [
            ValidationStatus.VALID,
            ValidationStatus.NOT_FOUND,
            ValidationStatus.UNCERTAIN,
        ]

    @pytest.mark.asyncio
    async def test_validate_method_call_valid(
        self,
        mock_validator,
        neo4j_query_responses,
    ):
        """Test validation of valid method calls"""
        await mock_validator.initialize()

        # Mock method call
        mock_method_call = MagicMock()
        mock_method_call.object_name = "test_obj"
        mock_method_call.method_name = "test_method"
        mock_method_call.class_name = "TestClass"
        mock_method_call.args = ["param1"]
        mock_method_call.kwargs = {}

        # Set up mock responses
        mock_validator.driver.session_data = neo4j_query_responses["find_method"]

        result = await mock_validator._validate_single_method_call(mock_method_call)

        assert result is not None
        assert result.validation.status in [
            ValidationStatus.VALID,
            ValidationStatus.NOT_FOUND,
            ValidationStatus.UNCERTAIN,
        ]

    @pytest.mark.asyncio
    async def test_validate_method_call_invalid(
        self,
        mock_validator,
        neo4j_query_responses,
    ):
        """Test validation of invalid method calls"""
        await mock_validator.initialize()

        # Mock invalid method call
        mock_method_call = MagicMock()
        mock_method_call.object_name = "test_obj"
        mock_method_call.method_name = "nonexistent_method"
        mock_method_call.class_name = "TestClass"
        mock_method_call.args = []
        mock_method_call.kwargs = {}

        # Set up empty response (method not found)
        mock_validator.driver.session_data = neo4j_query_responses["empty_result"]

        result = await mock_validator._validate_single_method_call(mock_method_call)

        assert result is not None
        assert result.validation.status == ValidationStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_validate_attribute_access_valid(
        self,
        mock_validator,
        neo4j_query_responses,
    ):
        """Test validation of valid attribute access"""
        await mock_validator.initialize()

        # Mock attribute access
        mock_attribute = MagicMock()
        mock_attribute.object_name = "test_obj"
        mock_attribute.attribute_name = "test_attr"
        mock_attribute.class_name = "TestClass"

        # Set up mock responses
        mock_validator.driver.session_data = neo4j_query_responses["find_attribute"]

        result = await mock_validator._validate_single_attribute_access(mock_attribute)

        assert result is not None
        assert result.validation.status in [
            ValidationStatus.VALID,
            ValidationStatus.NOT_FOUND,
        ]

    @pytest.mark.asyncio
    async def test_validate_function_call_valid(
        self,
        mock_validator,
        neo4j_query_responses,
    ):
        """Test validation of valid function calls"""
        await mock_validator.initialize()

        # Mock function call
        mock_function_call = MagicMock()
        mock_function_call.function_name = "test_function"
        mock_function_call.module = "main"
        mock_function_call.args = ["param1", "param2"]
        mock_function_call.kwargs = {}

        # Set up mock responses
        mock_validator.driver.session_data = neo4j_query_responses["find_function"]

        result = await mock_validator._validate_single_function_call(mock_function_call)

        assert result is not None
        assert result.validation.status in [
            ValidationStatus.VALID,
            ValidationStatus.NOT_FOUND,
            ValidationStatus.UNCERTAIN,
        ]

    @pytest.mark.asyncio
    async def test_parameter_validation_correct_types(self, mock_validator):
        """Test parameter validation with correct types"""
        await mock_validator.initialize()

        # Mock function/method parameters from knowledge graph
        expected_params = {"param1": "str", "param2": "int"}
        provided_args = ["hello", 42]
        provided_kwargs = {}

        result = mock_validator._validate_parameters(
            expected_params,
            provided_args,
            provided_kwargs,
        )

        assert result.status == ValidationStatus.VALID
        assert result.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_parameter_validation_wrong_types(self, mock_validator):
        """Test parameter validation with incorrect types"""
        await mock_validator.initialize()

        # Mock function/method parameters from knowledge graph
        expected_params = {"param1": "str", "param2": "int"}
        provided_args = [42, "hello"]  # Wrong types
        provided_kwargs = {}

        result = mock_validator._validate_parameters(
            expected_params,
            provided_args,
            provided_kwargs,
        )

        assert result.status == ValidationStatus.UNCERTAIN
        assert result.confidence < 0.8

    @pytest.mark.asyncio
    async def test_parameter_validation_missing_args(self, mock_validator):
        """Test parameter validation with missing arguments"""
        await mock_validator.initialize()

        # Mock function/method parameters from knowledge graph
        expected_params = {"param1": "str", "param2": "int"}
        provided_args = ["hello"]  # Missing second argument
        provided_kwargs = {}

        result = mock_validator._validate_parameters(
            expected_params,
            provided_args,
            provided_kwargs,
        )

        assert result.status == ValidationStatus.UNCERTAIN
        assert result.confidence < 0.8

    @pytest.mark.asyncio
    async def test_parameter_validation_with_kwargs(self, mock_validator):
        """Test parameter validation with keyword arguments"""
        await mock_validator.initialize()

        # Mock function/method parameters from knowledge graph
        expected_params = {"param1": "str", "param2": "int"}
        provided_args = ["hello"]
        provided_kwargs = {"param2": 42}

        result = mock_validator._validate_parameters(
            expected_params,
            provided_args,
            provided_kwargs,
        )

        assert result.status == ValidationStatus.VALID
        assert result.confidence >= 0.8

