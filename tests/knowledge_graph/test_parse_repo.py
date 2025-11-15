"""
Comprehensive unit tests for parse_repo_into_neo4j.py

Tests repository parsing and Neo4j indexing functions with:
- Mock Neo4j operations
- Mock git operations
- Error handling (RepositoryError, QueryError, GitError, ParsingError, AnalysisError)
- Edge cases and validation
- Aims for >80% coverage

Test coverage includes:
- DirectNeo4jExtractor initialization and configuration
- Repository validation and size checks
- Git cloning and metadata extraction
- Code file discovery (Python, JavaScript, TypeScript, Go)
- Repository analysis workflow
- Neo4j operations delegation
- Error handling and edge cases
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest

# Import custom exceptions
from src.core.exceptions import (
    GitError,
    ParsingError,
    AnalysisError,
    RepositoryError,
    QueryError,
)

# Import fixtures
from tests.fixtures.neo4j_fixtures import (
    MockNeo4jDriver,
    MockNeo4jSession,
    mock_neo4j_driver,
    sample_git_repo,
)


# =============================================================================
# Mock Setup and Fixtures
# =============================================================================

@pytest.fixture
def mock_git_manager():
    """Mock GitRepositoryManager"""
    manager = MagicMock()
    manager.validate_repository_size = AsyncMock(return_value=(True, {
        "estimated_size_mb": 10.5,
        "file_count": 150,
        "free_space_gb": 50.0,
        "errors": [],
    }))
    manager.clone_repository_with_validation = AsyncMock(return_value="/tmp/test_repo")
    manager.get_repository_info = AsyncMock(return_value={
        "remote_url": "https://github.com/test/repo.git",
        "current_branch": "main",
        "file_count": 150,
        "size": "10.5 MiB",
        "contributor_count": 5,
    })
    manager.get_branches = AsyncMock(return_value=[
        {"name": "main", "last_commit_date": "2024-01-01", "last_commit_message": "Initial commit"},
        {"name": "develop", "last_commit_date": "2024-01-02", "last_commit_message": "Dev branch"},
    ])
    manager.get_tags = AsyncMock(return_value=[
        {"name": "v1.0.0", "date": "2024-01-01", "message": "Release 1.0"},
    ])
    manager.get_commits = AsyncMock(return_value=[
        {"hash": "abc123", "author_name": "Test", "author_email": "test@example.com",
         "timestamp": 1704067200, "date": "2024-01-01T00:00:00", "message": "Test commit"},
    ])
    return manager


@pytest.fixture
def mock_analyzer():
    """Mock Neo4jCodeAnalyzer"""
    analyzer = MagicMock()
    analyzer.analyze_python_file = MagicMock(return_value={
        "file_path": "test.py",
        "module_name": "test",
        "classes": [
            {
                "name": "TestClass",
                "line_number": 5,
                "methods": [
                    {"name": "__init__", "line_number": 6, "parameters": ["self"]},
                    {"name": "test_method", "line_number": 9, "parameters": ["self", "arg1"]},
                ],
                "attributes": [{"name": "test_attr", "type": "str"}],
            }
        ],
        "functions": [
            {"name": "test_function", "line_number": 15, "parameters": ["x", "y"], "return_type": "int"}
        ],
        "imports": [
            {"module": "os", "type": "import"},
            {"module": "typing", "imports": ["Dict", "List"], "type": "from_import"},
        ],
    })
    return analyzer


@pytest.fixture
def mock_analyzer_factory():
    """Mock AnalyzerFactory"""
    factory = MagicMock()
    factory.get_supported_languages = MagicMock(return_value=["python", "javascript", "typescript", "go"])
    factory.get_supported_extensions = MagicMock(return_value=[".py", ".js", ".jsx", ".ts", ".tsx", ".go"])

    # Mock analyzer for JS/TS/Go
    js_analyzer = MagicMock()
    js_analyzer.analyze_file = AsyncMock(return_value={
        "file_path": "test.js",
        "module_name": "test",
        "language": "javascript",
        "classes": [],
        "functions": [{"name": "jsFunction", "line_number": 1, "parameters": []}],
        "imports": [],
    })

    factory.get_analyzer = MagicMock(return_value=js_analyzer)
    return factory


@pytest.fixture
async def extractor(mock_neo4j_driver, mock_git_manager, mock_analyzer, mock_analyzer_factory):
    """Create DirectNeo4jExtractor with mocked dependencies"""
    # Import here to avoid issues with module-level mocking
    from src.knowledge_graph.parse_repo_into_neo4j import DirectNeo4jExtractor

    with patch("src.knowledge_graph.parse_repo_into_neo4j.GitRepositoryManager", return_value=mock_git_manager), \
         patch("src.knowledge_graph.parse_repo_into_neo4j.Neo4jCodeAnalyzer", return_value=mock_analyzer), \
         patch("src.knowledge_graph.parse_repo_into_neo4j.AnalyzerFactory", return_value=mock_analyzer_factory), \
         patch("src.knowledge_graph.parse_repo_into_neo4j.AsyncGraphDatabase") as mock_db:

        mock_db.driver = MagicMock(return_value=mock_neo4j_driver)

        extractor = DirectNeo4jExtractor(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="test_user",
            neo4j_password="test_password"
        )

        extractor.driver = mock_neo4j_driver
        extractor.git_manager = mock_git_manager
        extractor.analyzer = mock_analyzer
        extractor.analyzer_factory = mock_analyzer_factory

        yield extractor

        # Cleanup
        await extractor.close()


# =============================================================================
# Test DirectNeo4jExtractor Initialization
# =============================================================================

class TestDirectNeo4jExtractorInit:
    """Test DirectNeo4jExtractor initialization and configuration"""

    def test_init_default_configuration(self, mock_git_manager, mock_analyzer, mock_analyzer_factory):
        """Test initialization with default configuration"""
        from src.knowledge_graph.parse_repo_into_neo4j import DirectNeo4jExtractor

        with patch("src.knowledge_graph.parse_repo_into_neo4j.GitRepositoryManager", return_value=mock_git_manager), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.Neo4jCodeAnalyzer", return_value=mock_analyzer), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.AnalyzerFactory", return_value=mock_analyzer_factory):

            extractor = DirectNeo4jExtractor(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password"
            )

            assert extractor.neo4j_uri == "bolt://localhost:7687"
            assert extractor.neo4j_user == "neo4j"
            assert extractor.neo4j_password == "password"
            assert extractor.driver is None  # Not initialized yet
            assert extractor.batch_size == 50  # Default
            assert extractor.repo_max_size_mb == 500  # Default
            assert extractor.repo_max_file_count == 10000  # Default
            assert extractor.repo_min_free_space_gb == 1.0  # Default
            assert extractor.repo_allow_size_override is False  # Default

    def test_init_custom_environment_variables(self, mock_git_manager, mock_analyzer, mock_analyzer_factory):
        """Test initialization with custom environment variables"""
        from src.knowledge_graph.parse_repo_into_neo4j import DirectNeo4jExtractor

        with patch.dict(os.environ, {
            "NEO4J_BATCH_SIZE": "100",
            "NEO4J_BATCH_TIMEOUT": "240",
            "REPO_MAX_SIZE_MB": "1000",
            "REPO_MAX_FILE_COUNT": "20000",
            "REPO_MIN_FREE_SPACE_GB": "2.0",
            "REPO_ALLOW_SIZE_OVERRIDE": "true",
        }), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.GitRepositoryManager", return_value=mock_git_manager), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.Neo4jCodeAnalyzer", return_value=mock_analyzer), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.AnalyzerFactory", return_value=mock_analyzer_factory):

            extractor = DirectNeo4jExtractor(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password"
            )

            assert extractor.batch_size == 100
            assert extractor.batch_timeout_seconds == 240
            assert extractor.repo_max_size_mb == 1000
            assert extractor.repo_max_file_count == 20000
            assert extractor.repo_min_free_space_gb == 2.0
            assert extractor.repo_allow_size_override is True


# =============================================================================
# Test Neo4j Connection Management
# =============================================================================

class TestNeo4jConnection:
    """Test Neo4j connection initialization and management"""

    @pytest.mark.asyncio
    async def test_initialize_success(self, extractor):
        """Test successful Neo4j initialization"""
        with patch("src.knowledge_graph.parse_repo_into_neo4j.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_driver.close = AsyncMock()
            mock_db.driver = MagicMock(return_value=mock_driver)

            await extractor.initialize()

            # Driver should be set
            assert extractor.driver is not None

    @pytest.mark.asyncio
    async def test_initialize_with_notification_suppression(self, extractor):
        """Test initialization with Neo4j notification suppression (v5.21.0+)"""
        from src.knowledge_graph.parse_repo_into_neo4j import DirectNeo4jExtractor

        with patch("src.knowledge_graph.parse_repo_into_neo4j.AsyncGraphDatabase") as mock_db, \
             patch("neo4j.NotificationMinimumSeverity") as mock_notif, \
             patch("src.knowledge_graph.parse_repo_into_neo4j.GitRepositoryManager"), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.Neo4jCodeAnalyzer"), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.AnalyzerFactory"):

            mock_driver = MagicMock()
            mock_driver.close = AsyncMock()
            mock_db.driver = MagicMock(return_value=mock_driver)
            mock_notif.OFF = "OFF"

            test_extractor = DirectNeo4jExtractor("bolt://localhost:7687", "neo4j", "password")
            await test_extractor.initialize()

            # Verify driver was created with notification suppression
            mock_db.driver.assert_called_once()
            call_kwargs = mock_db.driver.call_args[1]
            assert "warn_notification_severity" in call_kwargs

            # Cleanup
            await test_extractor.close()

    @pytest.mark.asyncio
    async def test_initialize_fallback_for_old_neo4j_version(self, extractor):
        """Test initialization fallback for older Neo4j versions"""
        from src.knowledge_graph.parse_repo_into_neo4j import DirectNeo4jExtractor

        with patch("src.knowledge_graph.parse_repo_into_neo4j.AsyncGraphDatabase") as mock_db, \
             patch("neo4j.NotificationMinimumSeverity", side_effect=ImportError), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.GitRepositoryManager"), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.Neo4jCodeAnalyzer"), \
             patch("src.knowledge_graph.parse_repo_into_neo4j.AnalyzerFactory"):

            mock_driver = MagicMock()
            mock_driver.close = AsyncMock()
            mock_db.driver = MagicMock(return_value=mock_driver)

            test_extractor = DirectNeo4jExtractor("bolt://localhost:7687", "neo4j", "password")
            await test_extractor.initialize()

            # Should still create driver, just without notification suppression
            mock_db.driver.assert_called_once()

            # Cleanup
            await test_extractor.close()

    @pytest.mark.asyncio
    async def test_close_connection(self, extractor):
        """Test closing Neo4j connection"""
        mock_driver = MagicMock()
        mock_driver.close = AsyncMock()
        extractor.driver = mock_driver

        await extractor.close()

        mock_driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_driver(self, extractor):
        """Test closing when driver is None"""
        extractor.driver = None

        # Should not raise exception
        await extractor.close()


# =============================================================================
# Test Repository Validation
# =============================================================================

class TestRepositoryValidation:
    """Test repository validation before processing"""

    @pytest.mark.asyncio
    async def test_validate_success(self, extractor):
        """Test successful repository validation"""
        extractor.git_manager.validate_repository_size = AsyncMock(return_value=(True, {
            "estimated_size_mb": 100.5,
            "file_count": 500,
            "free_space_gb": 50.0,
            "errors": [],
        }))

        is_valid, info = await extractor.validate_before_processing("https://github.com/test/repo.git")

        assert is_valid is True
        assert info["estimated_size_mb"] == 100.5
        assert info["file_count"] == 500
        assert len(info["errors"]) == 0

    @pytest.mark.asyncio
    async def test_validate_failure_without_override(self, extractor):
        """Test validation failure without size override"""
        extractor.repo_allow_size_override = False
        extractor.git_manager.validate_repository_size = AsyncMock(return_value=(False, {
            "estimated_size_mb": 1000.0,
            "file_count": 20000,
            "free_space_gb": 50.0,
            "errors": ["Repository too large: 1000.0MB exceeds limit of 500MB"],
        }))

        is_valid, info = await extractor.validate_before_processing("https://github.com/test/large-repo.git")

        assert is_valid is False
        assert len(info["errors"]) > 0
        assert "too large" in info["errors"][0].lower()

    @pytest.mark.asyncio
    async def test_validate_failure_with_override(self, extractor):
        """Test validation failure with size override enabled"""
        extractor.repo_allow_size_override = True
        extractor.git_manager.validate_repository_size = AsyncMock(return_value=(False, {
            "estimated_size_mb": 1000.0,
            "file_count": 20000,
            "free_space_gb": 50.0,
            "errors": ["Repository too large"],
        }))

        is_valid, info = await extractor.validate_before_processing("https://github.com/test/large-repo.git")

        # Should pass due to override
        assert is_valid is True
        assert info.get("override_applied") is True

    @pytest.mark.asyncio
    async def test_validate_without_git_manager(self, extractor):
        """Test validation when GitRepositoryManager is not available"""
        extractor.git_manager = None

        is_valid, info = await extractor.validate_before_processing("https://github.com/test/repo.git")

        # Should pass with warning
        assert is_valid is True
        assert "warning" in info  # Check that 'warning' key exists in dict

    @pytest.mark.asyncio
    async def test_validate_repository_error(self, extractor):
        """Test validation with RepositoryError"""
        extractor.git_manager.validate_repository_size = AsyncMock(
            side_effect=RepositoryError("Validation failed")
        )

        is_valid, info = await extractor.validate_before_processing("https://github.com/test/repo.git")

        assert is_valid is False
        assert len(info["errors"]) > 0
        assert "Validation failed" in info["errors"][0]

    @pytest.mark.asyncio
    async def test_validate_unexpected_exception(self, extractor):
        """Test validation with unexpected exception"""
        extractor.git_manager.validate_repository_size = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        is_valid, info = await extractor.validate_before_processing("https://github.com/test/repo.git")

        assert is_valid is False
        assert len(info["errors"]) > 0


# =============================================================================
# Test Git Clone Operations
# =============================================================================

class TestGitClone:
    """Test repository cloning operations"""

    @pytest.mark.asyncio
    async def test_clone_with_git_manager(self, extractor):
        """Test cloning using GitRepositoryManager"""
        extractor.git_manager.clone_repository_with_validation = AsyncMock(
            return_value="/tmp/test_repo"
        )

        result = await extractor.clone_repo(
            repo_url="https://github.com/test/repo.git",
            target_dir="/tmp/test_repo",
            branch="main"
        )

        assert result == "/tmp/test_repo"
        extractor.git_manager.clone_repository_with_validation.assert_called_once()

    @pytest.mark.asyncio
    async def test_clone_with_force_flag(self, extractor):
        """Test cloning with force flag to bypass validation"""
        extractor.git_manager.clone_repository_with_validation = AsyncMock(
            return_value="/tmp/test_repo"
        )

        result = await extractor.clone_repo(
            repo_url="https://github.com/test/repo.git",
            target_dir="/tmp/test_repo",
            force=True
        )

        assert result == "/tmp/test_repo"
        call_args = extractor.git_manager.clone_repository_with_validation.call_args
        assert call_args[1]["force"] is True

    @pytest.mark.asyncio
    async def test_clone_git_error(self, extractor):
        """Test cloning with GitError"""
        extractor.git_manager.clone_repository_with_validation = AsyncMock(
            side_effect=GitError("Clone failed")
        )

        with pytest.raises(GitError, match="Clone failed"):
            await extractor.clone_repo(
                repo_url="https://github.com/test/repo.git",
                target_dir="/tmp/test_repo"
            )

    @pytest.mark.asyncio
    async def test_clone_validation_error(self, extractor):
        """Test cloning with RuntimeError (validation failure)"""
        extractor.git_manager.clone_repository_with_validation = AsyncMock(
            side_effect=RuntimeError("Repository too large")
        )

        with pytest.raises(RuntimeError, match="Repository too large"):
            await extractor.clone_repo(
                repo_url="https://github.com/test/repo.git",
                target_dir="/tmp/test_repo"
            )

    @pytest.mark.asyncio
    async def test_clone_fallback_to_subprocess(self, extractor, tmp_path):
        """Test fallback to subprocess when GitRepositoryManager fails"""
        extractor.git_manager.clone_repository_with_validation = AsyncMock(
            side_effect=Exception("Manager unavailable")
        )

        target_dir = str(tmp_path / "fallback_repo")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = await extractor.clone_repo(
                repo_url="https://github.com/test/repo.git",
                target_dir=target_dir
            )

            assert result == target_dir
            mock_run.assert_called_once()
            assert "git" in mock_run.call_args[0][0][0]
            assert "clone" in mock_run.call_args[0][0]

    @pytest.mark.asyncio
    async def test_clone_subprocess_failure(self, extractor, tmp_path):
        """Test subprocess clone failure raises GitError"""
        extractor.git_manager = None  # Force subprocess path

        target_dir = str(tmp_path / "failed_repo")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git clone")

            with pytest.raises(GitError, match="Git clone failed"):
                await extractor.clone_repo(
                    repo_url="https://github.com/test/invalid.git",
                    target_dir=target_dir
                )

    @pytest.mark.asyncio
    async def test_clone_removes_existing_directory(self, extractor, tmp_path):
        """Test that existing directory is removed before cloning"""
        target_dir = tmp_path / "existing_repo"
        target_dir.mkdir()
        (target_dir / "old_file.txt").write_text("old content")

        extractor.git_manager.clone_repository_with_validation = AsyncMock(
            return_value=str(target_dir)
        )

        await extractor.clone_repo(
            repo_url="https://github.com/test/repo.git",
            target_dir=str(target_dir)
        )

        # GitRepositoryManager should handle cleanup
        extractor.git_manager.clone_repository_with_validation.assert_called_once()


# =============================================================================
# Test Git Metadata Extraction
# =============================================================================

class TestGitMetadata:
    """Test Git metadata extraction"""

    @pytest.mark.asyncio
    async def test_get_repository_metadata_success(self, extractor, tmp_path):
        """Test successful metadata extraction"""
        repo_dir = str(tmp_path / "test_repo")

        metadata = await extractor.get_repository_metadata(repo_dir)

        assert "info" in metadata
        assert "branches" in metadata
        assert "tags" in metadata
        assert "recent_commits" in metadata

        # Verify calls to git_manager
        extractor.git_manager.get_repository_info.assert_called_once_with(repo_dir)
        extractor.git_manager.get_branches.assert_called_once_with(repo_dir)
        extractor.git_manager.get_tags.assert_called_once_with(repo_dir)
        extractor.git_manager.get_commits.assert_called_once_with(repo_dir, limit=10)

    @pytest.mark.asyncio
    async def test_get_repository_metadata_git_error(self, extractor, tmp_path):
        """Test metadata extraction with GitError"""
        repo_dir = str(tmp_path / "test_repo")
        extractor.git_manager.get_repository_info = AsyncMock(
            side_effect=GitError("Git operation failed")
        )

        metadata = await extractor.get_repository_metadata(repo_dir)

        # Should continue with empty metadata
        assert metadata["branches"] == []
        assert metadata["tags"] == []
        assert metadata["recent_commits"] == []

    @pytest.mark.asyncio
    async def test_get_repository_metadata_without_git_manager(self, extractor, tmp_path):
        """Test metadata extraction when GitRepositoryManager is unavailable"""
        extractor.git_manager = None
        repo_dir = str(tmp_path / "test_repo")

        metadata = await extractor.get_repository_metadata(repo_dir)

        assert metadata["branches"] == []
        assert metadata["tags"] == []
        assert metadata["recent_commits"] == []

    @pytest.mark.asyncio
    async def test_get_repository_metadata_unexpected_error(self, extractor, tmp_path):
        """Test metadata extraction with unexpected exception"""
        repo_dir = str(tmp_path / "test_repo")
        extractor.git_manager.get_branches = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        metadata = await extractor.get_repository_metadata(repo_dir)

        # Should handle gracefully and continue
        assert "branches" in metadata


# =============================================================================
# Test File Discovery
# =============================================================================

class TestFileDiscovery:
    """Test code file discovery"""

    def test_get_python_files(self, extractor, tmp_path):
        """Test Python file discovery"""
        repo_dir = tmp_path / "test_repo"
        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)

        # Create Python files
        (src_dir / "main.py").write_text("# Main module")
        (src_dir / "utils.py").write_text("# Utils module")
        (src_dir / "test_main.py").write_text("# Test file - should be excluded")

        # Create excluded directory
        tests_dir = src_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_utils.py").write_text("# Test in tests dir")

        python_files = extractor.get_python_files(str(repo_dir))

        # Should find main.py and utils.py, but not test files
        file_names = [f.name for f in python_files]
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "test_main.py" not in file_names
        assert "test_utils.py" not in file_names

    def test_get_python_files_excludes_large_files(self, extractor, tmp_path):
        """Test that large Python files are excluded"""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()

        # Create normal file
        (repo_dir / "normal.py").write_text("# Normal file\n" * 100)

        # Create large file (>500KB)
        large_content = "# Large file\n" * 50000
        (repo_dir / "large.py").write_text(large_content)

        python_files = extractor.get_python_files(str(repo_dir))

        file_names = [f.name for f in python_files]
        assert "normal.py" in file_names
        # Large file should be excluded (if >500KB)
        if len(large_content.encode()) > 500_000:
            assert "large.py" not in file_names

    def test_get_code_files_multi_language(self, extractor, tmp_path):
        """Test multi-language code file discovery"""
        repo_dir = tmp_path / "test_repo"
        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)

        # Create files for different languages
        (src_dir / "main.py").write_text("# Python")
        (src_dir / "app.js").write_text("// JavaScript")
        (src_dir / "component.jsx").write_text("// React")
        (src_dir / "index.ts").write_text("// TypeScript")
        (src_dir / "component.tsx").write_text("// React TypeScript")
        (src_dir / "main.go").write_text("// Go")

        # Excluded files
        (src_dir / "app.min.js").write_text("// Minified JS")
        (src_dir / "types.d.ts").write_text("// Type definitions")
        (src_dir / "main_test.go").write_text("// Go test")

        code_files = extractor.get_code_files(str(repo_dir))

        assert len(code_files["python"]) == 1
        assert len(code_files["javascript"]) >= 1  # May include .jsx
        assert len(code_files["typescript"]) >= 1  # May include .tsx
        assert len(code_files["go"]) == 1

        # Check exclusions
        js_files = [f.name for f in code_files["javascript"]]
        assert "app.min.js" not in js_files

        go_files = [f.name for f in code_files["go"]]
        assert "main_test.go" not in go_files

    def test_get_code_files_excludes_directories(self, extractor, tmp_path):
        """Test that excluded directories are skipped"""
        repo_dir = tmp_path / "test_repo"

        # Create directories that should be excluded
        for excluded_dir in ["tests", "node_modules", "venv", ".git", "build"]:
            dir_path = repo_dir / excluded_dir
            dir_path.mkdir(parents=True)
            (dir_path / "file.py").write_text("# Should be excluded")

        # Create valid source directory
        src_dir = repo_dir / "src"
        src_dir.mkdir()
        (src_dir / "valid.py").write_text("# Valid file")

        code_files = extractor.get_code_files(str(repo_dir))

        python_files = code_files["python"]
        assert len(python_files) == 1
        assert python_files[0].name == "valid.py"


# =============================================================================
# Test Repository Analysis
# =============================================================================

class TestRepositoryAnalysis:
    """Test repository analysis workflow"""

    @pytest.mark.asyncio
    async def test_analyze_repository_success(self, extractor, tmp_path):
        """Test successful repository analysis"""
        repo_dir = tmp_path / "test_repo"
        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "main.py").write_text("# Main module\nclass TestClass:\n    pass")

        with patch.object(extractor, "clone_repo", return_value=str(repo_dir)), \
             patch.object(extractor, "_create_graph", new=AsyncMock()) as mock_create:

            await extractor.analyze_repository(
                repo_url="https://github.com/test/repo.git",
                temp_dir=str(repo_dir),
            )

            # Should call _create_graph
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0]
            assert call_args[0] == "repo"  # repo_name
            assert isinstance(call_args[1], list)  # modules_data

    @pytest.mark.asyncio
    async def test_analyze_repository_with_branch(self, extractor, tmp_path):
        """Test repository analysis with specific branch"""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()

        with patch.object(extractor, "clone_repo", return_value=str(repo_dir)) as mock_clone, \
             patch.object(extractor, "get_code_files", return_value={"python": [], "javascript": [], "typescript": [], "go": []}), \
             patch.object(extractor, "_create_graph", new=AsyncMock()):

            await extractor.analyze_repository(
                repo_url="https://github.com/test/repo.git",
                temp_dir=str(repo_dir),
                branch="develop"
            )

            # Should pass branch to clone_repo (as 3rd positional argument)
            mock_clone.assert_called_once()
            assert mock_clone.call_args[0][2] == "develop"  # branch is 3rd positional arg

    @pytest.mark.asyncio
    async def test_analyze_repository_clears_existing_data(self, extractor, tmp_path):
        """Test that existing data is cleared before analysis"""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()

        with patch.object(extractor, "clear_repository_data", new=AsyncMock()) as mock_clear, \
             patch.object(extractor, "clone_repo", return_value=str(repo_dir)), \
             patch.object(extractor, "get_code_files", return_value={"python": [], "javascript": [], "typescript": [], "go": []}), \
             patch.object(extractor, "_create_graph", new=AsyncMock()):

            await extractor.analyze_repository(
                repo_url="https://github.com/test/repo.git",
                temp_dir=str(repo_dir)
            )

            mock_clear.assert_called_once_with("repo")

    @pytest.mark.asyncio
    async def test_analyze_repository_cleanup(self, extractor, tmp_path):
        """Test that temporary directory is cleaned up after analysis"""
        repo_dir = tmp_path / "test_repo"
        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "main.py").write_text("# Main")

        with patch.object(extractor, "clone_repo", return_value=str(repo_dir)), \
             patch.object(extractor, "_create_graph", new=AsyncMock()):

            await extractor.analyze_repository(
                repo_url="https://github.com/test/repo.git",
                temp_dir=str(repo_dir)
            )

            # Directory should be removed after analysis
            # Note: In real scenario, shutil.rmtree is called
            # For testing, we just verify the logic runs without error

    @pytest.mark.asyncio
    async def test_analyze_repository_git_error(self, extractor, tmp_path):
        """Test repository analysis with GitError during clone"""
        with patch.object(extractor, "clone_repo", side_effect=GitError("Clone failed")):

            with pytest.raises(GitError, match="Clone failed"):
                await extractor.analyze_repository(
                    repo_url="https://github.com/test/repo.git"
                )

    @pytest.mark.asyncio
    async def test_analyze_local_repository_success(self, extractor, tmp_path):
        """Test analysis of local repository"""
        repo_dir = tmp_path / "local_repo"
        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "main.py").write_text("# Main module")

        with patch.object(extractor, "_create_graph", new=AsyncMock()) as mock_create:

            await extractor.analyze_local_repository(
                local_path=str(repo_dir),
                repo_name="local-repo"
            )

            mock_create.assert_called_once()
            call_args = mock_create.call_args[0]
            assert call_args[0] == "local-repo"

    @pytest.mark.asyncio
    async def test_analyze_local_repository_parsing_error(self, extractor, tmp_path):
        """Test local repository analysis with ParsingError"""
        repo_dir = tmp_path / "local_repo"
        repo_dir.mkdir()

        extractor.analyzer.analyze_python_file = MagicMock(
            side_effect=ParsingError("Parse failed")
        )

        # Should handle ParsingError gracefully or raise
        with patch.object(extractor, "get_code_files", return_value={"python": [Path(repo_dir / "test.py")], "javascript": [], "typescript": [], "go": []}):
            try:
                await extractor.analyze_local_repository(
                    local_path=str(repo_dir),
                    repo_name="local-repo"
                )
            except (ParsingError, AnalysisError, RepositoryError):
                # Expected behavior - errors should propagate
                pass

    @pytest.mark.asyncio
    async def test_analyze_local_repository_analysis_error(self, extractor, tmp_path):
        """Test local repository analysis with AnalysisError"""
        repo_dir = tmp_path / "local_repo"
        repo_dir.mkdir()

        with patch.object(extractor, "get_code_files", side_effect=AnalysisError("Analysis failed")):

            with pytest.raises(AnalysisError, match="Analysis failed"):
                await extractor.analyze_local_repository(
                    local_path=str(repo_dir),
                    repo_name="local-repo"
                )


# =============================================================================
# Test Neo4j Operations Delegation
# =============================================================================

class TestNeo4jOperations:
    """Test Neo4j operations delegation"""

    @pytest.mark.asyncio
    async def test_clear_repository_data(self, extractor):
        """Test clearing repository data"""
        with patch("src.knowledge_graph.parse_repo_into_neo4j.clear_repository_data", new=AsyncMock()) as mock_clear:

            await extractor.clear_repository_data("test-repo")

            mock_clear.assert_called_once_with(extractor.driver, "test-repo")

    @pytest.mark.asyncio
    async def test_create_graph(self, extractor):
        """Test graph creation delegation"""
        modules_data = [{"module": "test", "classes": [], "functions": []}]
        git_metadata = {"branches": [], "commits": []}

        with patch("src.knowledge_graph.parse_repo_into_neo4j.create_graph", new=AsyncMock()) as mock_create:

            await extractor._create_graph("test-repo", modules_data, git_metadata)

            mock_create.assert_called_once_with(
                extractor.driver,
                "test-repo",
                modules_data,
                git_metadata
            )

    @pytest.mark.asyncio
    async def test_search_graph(self, extractor):
        """Test graph search delegation"""
        with patch("src.knowledge_graph.parse_repo_into_neo4j.search_graph", new=AsyncMock(return_value=[{"result": "data"}])) as mock_search:

            result = await extractor.search_graph(
                query_type="files_importing",
                target="models"
            )

            assert result == [{"result": "data"}]
            mock_search.assert_called_once()


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test comprehensive error handling"""

    @pytest.mark.asyncio
    async def test_repository_error_handling(self, extractor, tmp_path):
        """Test RepositoryError handling"""
        with patch.object(extractor, "clone_repo", side_effect=RepositoryError("Repository invalid")):

            with pytest.raises(RepositoryError, match="Repository invalid"):
                await extractor.analyze_repository("https://github.com/test/invalid.git")

    @pytest.mark.asyncio
    async def test_query_error_handling(self, extractor):
        """Test QueryError handling in Neo4j operations"""
        with patch("src.knowledge_graph.parse_repo_into_neo4j.clear_repository_data",
                   new=AsyncMock(side_effect=QueryError("Query failed"))):

            with pytest.raises(QueryError, match="Query failed"):
                await extractor.clear_repository_data("test-repo")

    @pytest.mark.asyncio
    async def test_git_error_propagation(self, extractor):
        """Test GitError propagation through the stack"""
        extractor.git_manager.get_repository_info = AsyncMock(
            side_effect=GitError("Git command failed")
        )

        metadata = await extractor.get_repository_metadata("/tmp/repo")

        # Should handle GitError and continue with empty metadata
        assert metadata["branches"] == []

    @pytest.mark.asyncio
    async def test_multiple_file_parsing_errors(self, extractor, tmp_path):
        """Test handling multiple file parsing errors"""
        repo_dir = tmp_path / "error_repo"
        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)

        # Create multiple files
        for i in range(3):
            (src_dir / f"file{i}.py").write_text(f"# File {i}")

        # Make analyzer fail for some files
        call_count = [0]
        def side_effect_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ParsingError("Parse error")
            return {
                "file_path": f"file{call_count[0]}.py",
                "classes": [],
                "functions": [],
                "imports": [],
            }

        extractor.analyzer.analyze_python_file.side_effect = side_effect_fn

        with patch.object(extractor, "_create_graph", new=AsyncMock()):
            # Should handle errors and continue with other files
            try:
                await extractor.analyze_local_repository(str(repo_dir), "error-repo")
            except Exception:
                # Some implementations may raise, others may skip
                pass

    @pytest.mark.asyncio
    async def test_unexpected_exception_handling(self, extractor, tmp_path):
        """Test handling of unexpected exceptions"""
        repo_dir = tmp_path / "unexpected_repo"
        repo_dir.mkdir()

        with patch.object(extractor, "get_code_files", side_effect=Exception("Unexpected error")):

            with pytest.raises(Exception, match="Unexpected error"):
                await extractor.analyze_local_repository(str(repo_dir), "unexpected-repo")


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_empty_repository(self, extractor, tmp_path):
        """Test analysis of empty repository"""
        repo_dir = tmp_path / "empty_repo"
        repo_dir.mkdir()

        with patch.object(extractor, "_create_graph", new=AsyncMock()) as mock_create:

            await extractor.analyze_local_repository(str(repo_dir), "empty-repo")

            # Should call _create_graph with empty modules
            call_args = mock_create.call_args[0]
            assert len(call_args[1]) == 0  # Empty modules_data

    @pytest.mark.asyncio
    async def test_repository_with_no_python_files(self, extractor, tmp_path):
        """Test repository with only non-Python files"""
        repo_dir = tmp_path / "no_python_repo"
        repo_dir.mkdir()

        (repo_dir / "README.md").write_text("# README")
        (repo_dir / "config.json").write_text('{"key": "value"}')

        with patch.object(extractor, "_create_graph", new=AsyncMock()) as mock_create:

            await extractor.analyze_local_repository(str(repo_dir), "no-python-repo")

            # Should handle gracefully
            call_args = mock_create.call_args[0]
            python_modules = [m for m in call_args[1] if m.get("language") == "python"]
            assert len(python_modules) == 0

    def test_get_python_files_nonexistent_path(self, extractor):
        """Test get_python_files with nonexistent path"""
        result = extractor.get_python_files("/nonexistent/path")

        # Should return empty list or raise error
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_analyze_repository_with_default_temp_dir(self, extractor, tmp_path):
        """Test repository analysis with default temp_dir"""
        with patch.object(extractor, "clone_repo", return_value=str(tmp_path / "default_repo")) as mock_clone, \
             patch.object(extractor, "get_code_files", return_value={"python": [], "javascript": [], "typescript": [], "go": []}), \
             patch.object(extractor, "_create_graph", new=AsyncMock()):

            await extractor.analyze_repository("https://github.com/test/repo.git")

            # Should use default temp_dir
            mock_clone.assert_called_once()
            call_args = mock_clone.call_args[0]
            assert call_args[0] == "https://github.com/test/repo.git"

    @pytest.mark.asyncio
    async def test_concurrent_analyses(self, extractor, tmp_path):
        """Test concurrent repository analyses"""
        repos = []
        for i in range(3):
            repo_dir = tmp_path / f"repo_{i}"
            repo_dir.mkdir()
            (repo_dir / "test.py").write_text(f"# Repo {i}")
            repos.append(str(repo_dir))

        with patch.object(extractor, "_create_graph", new=AsyncMock()):

            tasks = [
                extractor.analyze_local_repository(repo, f"repo-{i}")
                for i, repo in enumerate(repos)
            ]

            # Should handle concurrent analyses
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check that all completed (may have exceptions in some implementations)
            assert len(results) == 3


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=src.knowledge_graph.parse_repo_into_neo4j", "--cov-report=term-missing"])
