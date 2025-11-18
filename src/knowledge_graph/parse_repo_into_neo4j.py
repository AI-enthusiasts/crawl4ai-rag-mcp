"""
Direct Neo4j GitHub Code Repository Extractor

Creates nodes and relationships directly in Neo4j without Graphiti:
- File nodes
- Class nodes
- Method nodes
- Function nodes
- Import relationships

Bypasses all LLM processing for maximum speed.
"""

import asyncio
import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import AsyncDriver, AsyncGraphDatabase

# Handle neo4j version compatibility for NotificationMinimumSeverity
try:
    from neo4j import NotificationMinimumSeverity
except ImportError:
    NotificationMinimumSeverity = None

from src.core.exceptions import AnalysisError, GitError, ParsingError, RepositoryError

# Import analyzer components for multi-language support
from .analyzers import Neo4jCodeAnalyzer
from .analyzers.factory import AnalyzerFactory

# Import GitRepositoryManager from same package
from .git_manager import GitRepositoryManager

# Import Neo4j operations from neo4j package
from .neo4j import clear_repository_data, create_graph, search_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DirectNeo4jExtractor:
    """Creates nodes and relationships directly in Neo4j"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver: AsyncDriver | None = None
        self.analyzer = Neo4jCodeAnalyzer()
        # Initialize GitRepositoryManager for Git metadata collection
        self.git_manager: GitRepositoryManager = GitRepositoryManager()
        # Initialize analyzer factory for multi-language support
        self.analyzer_factory = AnalyzerFactory()
        # Transaction batching configuration
        self.batch_size = int(os.environ.get("NEO4J_BATCH_SIZE", "50"))
        self.batch_timeout_seconds = int(os.environ.get("NEO4J_BATCH_TIMEOUT", "120"))

        # Repository size limits from environment
        self.repo_max_size_mb = int(os.environ.get("REPO_MAX_SIZE_MB", "500"))
        self.repo_max_file_count = int(os.environ.get("REPO_MAX_FILE_COUNT", "10000"))
        self.repo_min_free_space_gb = float(
            os.environ.get("REPO_MIN_FREE_SPACE_GB", "1.0"),
        )
        allow_override_str = os.environ.get("REPO_ALLOW_SIZE_OVERRIDE", "false")
        self.repo_allow_size_override = allow_override_str.lower() == "true"

        logger.info(
            "Git metadata collection enabled with GitRepositoryManager",
        )
        supported_langs = self.analyzer_factory.get_supported_languages()
        logger.info(
            "Multi-language support enabled for: %s",
            ", ".join(supported_langs),
        )
        logger.info(
            "Repository limits - Max size: %sMB, Max files: %s, "
            "Min free space: %sGB, Allow override: %s",
            self.repo_max_size_mb,
            self.repo_max_file_count,
            self.repo_min_free_space_gb,
            self.repo_allow_size_override,
        )

    async def initialize(self) -> None:
        """Initialize Neo4j connection"""
        logger.info("Initializing Neo4j connection...")

        # Create Neo4j driver with notification suppression if available
        driver_kwargs: dict[str, Any] = {
            "auth": (self.neo4j_user, self.neo4j_password),
        }
        if NotificationMinimumSeverity is not None:
            driver_kwargs["warn_notification_severity"] = (
                NotificationMinimumSeverity.OFF
            )
        else:
            # Fallback for older versions - use logging suppression
            logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

        self.driver = AsyncGraphDatabase.driver(self.neo4j_uri, **driver_kwargs)

        logger.info("Neo4j connection initialized successfully")

    async def clear_repository_data(self, repo_name: str) -> None:
        """Delegate to neo4j.clear_repository_data"""
        await clear_repository_data(self.driver, repo_name)

    async def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()

    async def validate_before_processing(
        self, repo_url: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate repository before processing.

        Args:
            repo_url: Repository URL to validate

        Returns:
            Tuple of (is_valid, info) with validation details
        """
        if not self.git_manager:
            msg = "GitRepositoryManager not available, skipping validation"
            logger.warning(msg)
            return True, {"warning": "Validation skipped - " + msg}

        try:
            is_valid, info = await self.git_manager.validate_repository_size(
                url=repo_url,
                max_size_mb=self.repo_max_size_mb,
                max_file_count=self.repo_max_file_count,
                min_free_space_gb=self.repo_min_free_space_gb,
            )

            if is_valid:
                return is_valid, info
            elif self.repo_allow_size_override:
                logger.warning(
                    "Repository exceeds limits but override is enabled: %s",
                    info.get("errors", []),
                )
                info["override_applied"] = True
                return True, info
            else:
                logger.error(
                    "Repository validation failed: %s",
                    info.get("errors", []),
                )
                return False, info
        except RepositoryError as e:
            logger.exception("Repository validation failed")
            return False, {"errors": [str(e)]}
        except Exception:
            logger.exception("Unexpected error during validation")
            return False, {"errors": ["Unexpected error"]}

    async def clone_repo(
        self,
        repo_url: str,
        target_dir: str,
        *,
        branch: str | None = None,
        force: bool = False,
    ) -> str:
        """Clone repository with size validation and enhanced Git support

        Args:
            repo_url: Repository URL to clone
            target_dir: Target directory for cloning
            branch: Optional branch to clone
            force: Force clone even if size validation fails

        Returns:
            Path to the cloned repository

        Raises:
            RuntimeError: If validation fails (unless force=True)
        """
        logger.info("Cloning repository to: %s", target_dir)

        # Use GitRepositoryManager with validation if available
        if self.git_manager:
            try:
                # Use the new validation method
                return await self.git_manager.clone_repository_with_validation(
                    url=repo_url,
                    target_dir=target_dir,
                    branch=branch,
                    depth=1,
                    single_branch=bool(branch),
                    max_size_mb=self.repo_max_size_mb,
                    max_file_count=self.repo_max_file_count,
                    min_free_space_gb=self.repo_min_free_space_gb,
                    force=force or self.repo_allow_size_override,
                )
            except RuntimeError:
                # Re-raise validation errors
                raise
            except GitError:
                logger.exception("Git operation failed")
                raise
            except Exception as e:
                logger.warning(
                    "GitRepositoryManager failed, falling back to subprocess: %s",
                    e,
                )

        # Fallback to original implementation
        if Path(target_dir).exists():
            logger.info("Removing existing directory: %s", target_dir)
            try:

                def handle_remove_readonly(
                    func: Callable[[str], None], path: str,
                ) -> None:
                    try:
                        if Path(path).exists():
                            Path(path).chmod(0o777)
                            func(path)
                    except PermissionError:
                        logger.warning(
                            "Could not remove %s - file in use, skipping", path,
                        )

                shutil.rmtree(target_dir, onerror=handle_remove_readonly)
            except Exception as e:
                logger.warning(
                    "Could not fully remove %s: %s. Proceeding anyway...",
                    target_dir,
                    e,
                )

        logger.info("Running git clone from %s", repo_url)
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_url, target_dir])

        try:
            subprocess.run(cmd, check=True)  # noqa: S603
            logger.info("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            msg = f"Git clone failed: {e}"
            logger.exception(msg)
            raise GitError(msg) from e
        else:
            return target_dir

    async def get_repository_metadata(self, repo_dir: str) -> dict[str, Any]:
        """Extract Git repository metadata using GitRepositoryManager"""
        metadata: dict[str, Any] = {
            "branches": [],
            "tags": [],
            "recent_commits": [],
            "info": {},
        }

        if self.git_manager:
            try:
                logger.info("Extracting Git metadata from %s", repo_dir)

                # Get repository info
                repo_info = await self.git_manager.get_repository_info(repo_dir)
                metadata["info"] = repo_info
                logger.debug("Repository info: %s", metadata["info"])

                # Get branches
                branches = await self.git_manager.get_branches(repo_dir)
                metadata["branches"] = branches
                logger.debug("Found %s branches", len(metadata["branches"]))

                # Get tags
                tags = await self.git_manager.get_tags(repo_dir)
                metadata["tags"] = tags
                logger.debug("Found %s tags", len(metadata["tags"]))

                # Get recent commits (last 10)
                commits = await self.git_manager.get_commits(repo_dir, limit=10)
                metadata["recent_commits"] = commits
                logger.debug("Found %s recent commits", len(metadata["recent_commits"]))

                logger.info(
                    "Successfully extracted Git metadata: "
                    "%s branches, %s tags, %s commits",
                    len(metadata["branches"]),
                    len(metadata["tags"]),
                    len(metadata["recent_commits"]),
                )
            except GitError:
                logger.exception("Git operation failed during metadata extraction")
                logger.warning("Continuing without Git metadata")
            except Exception:
                logger.exception("Unexpected error extracting Git metadata")
                logger.warning("Continuing without Git metadata")
        else:
            msg = (
                "GitRepositoryManager not available - "
                "skipping Git metadata extraction"
            )
            logger.warning(msg)

        return metadata

    def get_python_files(self, repo_path: str) -> list[Path]:
        """Get Python files, focusing on main source directories"""
        max_file_size = 500_000
        exclude_dirs = {
            "tests",
            "test",
            "__pycache__",
            ".git",
            "venv",
            "env",
            "node_modules",
            "build",
            "dist",
            ".pytest_cache",
            "docs",
            "examples",
            "example",
            "demo",
            "benchmark",
        }
        skip_files = {"setup.py", "conftest.py"}
        python_files = []

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [
                d
                for d in dirs
                if d not in exclude_dirs and not d.startswith(".")
            ]

            for file in files:
                if not (file.endswith(".py") and not file.startswith("test_")):
                    continue
                file_path = Path(root) / file
                if (
                    file_path.stat().st_size < max_file_size
                    and file not in skip_files
                ):
                    python_files.append(file_path)

        return python_files

    def get_code_files(self, repo_path: str) -> dict[str, list[Path]]:
        """Get all supported code files, organized by language"""
        max_file_size = 500_000
        exclude_dirs = {
            "tests",
            "test",
            "__pycache__",
            ".git",
            "venv",
            "env",
            "node_modules",
            "build",
            "dist",
            ".pytest_cache",
            "docs",
            "examples",
            "example",
            "demo",
            "benchmark",
            "vendor",
            ".next",
            ".nuxt",
            "coverage",
            "lib",
            "out",
        }

        py_skip_patterns = {"migrations/", "pb2.py", "_pb2_grpc.py"}
        js_skip_patterns = {".min.js", ".bundle.js", "webpack"}
        ts_skip_patterns = {".d.ts", ".min.js"}
        go_skip_patterns = {"_test.go", ".pb.go"}

        code_files: dict[str, list[Path]] = {
            "python": [],
            "javascript": [],
            "typescript": [],
            "go": [],
        }

        supported_extensions = (
            self.analyzer_factory.get_supported_extensions()
        )

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [
                d
                for d in dirs
                if d not in exclude_dirs and not d.startswith(".")
            ]

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                # Skip if not a supported extension or test file
                if ext not in supported_extensions:
                    continue
                if file.startswith("test_"):
                    continue
                if file.endswith((".test" + ext, ".spec" + ext)):
                    continue
                if file_path.stat().st_size >= max_file_size:
                    continue

                # Categorize by language
                file_str = str(file_path)
                if ext == ".py" and not any(
                    skip in file_str for skip in py_skip_patterns
                ):
                    code_files["python"].append(file_path)
                elif ext in {".js", ".jsx", ".mjs", ".cjs"} and not any(
                    skip in file_str for skip in js_skip_patterns
                ):
                    code_files["javascript"].append(file_path)
                elif ext in {".ts", ".tsx"} and not any(
                    skip in file_str for skip in ts_skip_patterns
                ):
                    code_files["typescript"].append(file_path)
                elif ext == ".go" and not any(
                    skip in file_str for skip in go_skip_patterns
                ):
                    code_files["go"].append(file_path)

        return code_files

    async def analyze_repository(
        self,
        repo_url: str,
        *,
        temp_dir: str | None = None,
        branch: str | None = None,
        force: bool = False,
    ) -> None:
        """Analyze repository and create nodes/relationships in Neo4j

        Args:
            repo_url: Repository URL to analyze
            temp_dir: Optional temporary directory for cloning
            branch: Optional branch to analyze
            force: Force analysis even if repository exceeds size limits

        Raises:
            RuntimeError: If repository validation fails (unless force=True)
        """
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        logger.info("Analyzing repository: %s", repo_name)

        await self.clear_repository_data(repo_name)

        if temp_dir is None:
            script_dir = Path(__file__).parent
            temp_dir = str(script_dir / "repos" / repo_name)

        repo_path = Path(
            await self.clone_repo(repo_url, temp_dir, branch=branch, force=force),
        )

        try:
            logger.info("Getting code files for all supported languages...")
            code_files = self.get_code_files(str(repo_path))

            total_files = sum(len(files) for files in code_files.values())
            logger.info("Found %s code files to analyze:", total_files)
            for lang, files in code_files.items():
                if files:
                    logger.info("  - %s: %s files", lang, len(files))

            logger.info("Identifying project modules...")
            project_modules = set()
            for file_path in code_files.get("python", []):
                relative_path = str(file_path.relative_to(repo_path))
                path_parts = (
                    relative_path.replace("/", ".").replace(".py", "").split(".")
                )
                if path_parts and not path_parts[0].startswith("."):
                    project_modules.add(path_parts[0])

            if project_modules:
                logger.info(
                    "Identified Python project modules: %s",
                    sorted(project_modules),
                )

            modules_data = []
            file_counter = 0

            for file_path in code_files.get("python", []):
                if file_counter % 20 == 0:
                    logger.info(
                        "Analyzing file %s/%s: %s",
                        file_counter + 1,
                        total_files,
                        file_path.name,
                    )
                file_counter += 1

                analysis = self.analyzer.analyze_python_file(
                    file_path, repo_path, project_modules,
                )
                if analysis:
                    analysis["language"] = "Python"
                    modules_data.append(analysis)

            js_analyzer = self.analyzer_factory.get_analyzer(".js")
            for lang, _ext in [("javascript", ".js"), ("typescript", ".ts")]:
                for file_path in code_files.get(lang, []):
                    if file_counter % 20 == 0:
                        logger.info(
                            "Analyzing file %s/%s: %s",
                            file_counter + 1,
                            total_files,
                            file_path.name,
                        )
                    file_counter += 1

                    if js_analyzer:
                        analysis = await js_analyzer.analyze_file(
                            str(file_path), str(repo_path),
                        )
                        if analysis:
                            modules_data.append(analysis)

            go_analyzer = self.analyzer_factory.get_analyzer(".go")
            for file_path in code_files.get("go", []):
                if file_counter % 20 == 0:
                    logger.info(
                        "Analyzing file %s/%s: %s",
                        file_counter + 1,
                        total_files,
                        file_path.name,
                    )
                file_counter += 1

                if go_analyzer:
                    analysis = await go_analyzer.analyze_file(
                        str(file_path), str(repo_path),
                    )
                    if analysis:
                        modules_data.append(analysis)

            logger.info("Found %s files with content", len(modules_data))

            git_metadata = await self.get_repository_metadata(str(repo_path))

            logger.info("Creating nodes and relationships in Neo4j...")
            await self._create_graph(repo_name, modules_data, git_metadata)

            total_classes = sum(
                len(mod["classes"]) for mod in modules_data
            )
            total_methods = sum(
                len(cls["methods"])
                for mod in modules_data
                for cls in mod["classes"]
            )
            total_functions = sum(
                len(mod["functions"]) for mod in modules_data
            )
            total_imports = sum(
                len(mod["imports"]) for mod in modules_data
            )

            logger.info(
                "\n=== Direct Neo4j Repository Analysis for %s ===", repo_name,
            )
            logger.info("Files processed: %s", len(modules_data))
            logger.info("Classes created: %s", total_classes)
            logger.info("Methods created: %s", total_methods)
            logger.info("Functions created: %s", total_functions)
            logger.info("Import relationships: %s", total_imports)

            logger.info("Successfully created Neo4j graph for %s", repo_name)

        finally:
            if Path(temp_dir).exists():
                logger.info("Cleaning up temporary directory: %s", temp_dir)
                try:

                    def handle_remove_readonly(
                        func: Callable[[str], None], path: str,
                    ) -> None:
                        try:
                            if Path(path).exists():
                                Path(path).chmod(0o777)
                                func(path)
                        except PermissionError:
                            logger.warning(
                                "Could not remove %s - file in use, skipping",
                                path,
                            )

                    shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
                    logger.info("Cleanup completed")
                except Exception as e:
                    logger.warning(
                        "Cleanup failed: %s. Directory may remain at %s",
                        e,
                        temp_dir,
                    )
                except GitError:
                    logger.exception("Git-related cleanup failed")


    async def analyze_local_repository(self, local_path: str, repo_name: str) -> None:
        """
        Analyze a local Git repository without cloning.

        Args:
            local_path: Absolute path to the local repository
            repo_name: Repository name for Neo4j storage
        """
        logger.info("Analyzing local repository: %s at %s", repo_name, local_path)

        # Clear existing data for this repository before re-processing
        await self.clear_repository_data(repo_name)

        repo_path = Path(local_path)

        try:
            # Get all code files, organized by language
            logger.info("Getting code files for all supported languages...")
            code_files = self.get_code_files(str(repo_path))

            total_files = sum(len(files) for files in code_files.values())
            logger.info("Found %s code files to analyze:", total_files)
            for lang, files in code_files.items():
                if files:
                    logger.info("  - %s: %s files", lang, len(files))

            logger.info("Identifying project modules...")
            project_modules = set()
            for file_path in code_files.get("python", []):
                relative = str(Path(file_path).relative_to(repo_path))
                # Convert file path to module name
                module_path = relative.replace("/", ".").replace(".py", "")
                if (
                    module_path
                    and not module_path.startswith("test")
                    and "__pycache__" not in module_path
                ):
                    project_modules.add(module_path.split(".")[0])

            logger.info(
                "Identified %s project modules: %s",
                len(project_modules),
                sorted(project_modules),
            )

            # Second pass: analyze all files
            logger.info("Analyzing code structure...")
            all_modules = []

            for lang, files in code_files.items():
                if not files:
                    continue

                logger.info("Analyzing %s %s files...", len(files), lang)
                analyzer = self.analyzer_factory.get_analyzer(lang)

                for file_path in files:
                    try:
                        file_path_obj = Path(file_path)
                        rel_path = file_path_obj.relative_to(repo_path)
                        logger.debug("Processing %s file: %s", lang, rel_path)

                        if lang == "python":
                            module_data = self.analyzer.analyze_python_file(
                                file_path_obj, repo_path, project_modules,
                            )
                        else:
                            module_data = (
                                await analyzer.analyze_file(
                                    str(file_path_obj), str(repo_path),
                                )
                                if analyzer
                                else None
                            )

                        if module_data:
                            module_data["language"] = lang
                            all_modules.append(module_data)

                    except Exception as e:
                        logger.warning("Failed to analyze %s: %s", file_path, e)
                        continue

            logger.info("Successfully analyzed %s files", len(all_modules))

            git_metadata: dict[str, Any] = {}
            if self.git_manager:
                try:
                    logger.info("Collecting Git metadata...")

                    repo_info = await self.git_manager.get_repository_info(
                        local_path,
                    )
                    branches = await self.git_manager.get_branches(local_path)
                    commits = await self.git_manager.get_commits(
                        local_path, limit=50,
                    )

                    git_metadata = {
                        "info": repo_info,
                        "branches": branches,
                        "commits": commits,
                    }

                    logger.info(
                        "Collected metadata: %s branches, %s commits",
                        len(branches),
                        len(commits),
                    )

                except GitError:
                    logger.exception(
                        "Git operation failed during metadata collection",
                    )
                except Exception:
                    logger.exception(
                        "Unexpected error collecting Git metadata",
                    )

            logger.info("Creating Neo4j graph...")
            await self._create_graph(repo_name, all_modules, git_metadata)

            logger.info("Analysis complete for local repository: %s", repo_name)

        except (GitError, ParsingError, AnalysisError, RepositoryError):
            logger.exception("Analysis failed for local repository")
            raise
        except Exception:
            logger.exception("Unexpected error analyzing local repository")
            raise
    async def _create_graph(
        self,
        repo_name: str,
        modules_data: list[dict[str, Any]],
        git_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to neo4j.create_graph"""
        await create_graph(self.driver, repo_name, modules_data, git_metadata)

    async def search_graph(
        self, query_type: str, **kwargs: Any,
    ) -> list[dict[str, Any]] | None:
        """Delegate to neo4j.search_graph"""
        return await search_graph(self.driver, query_type, **kwargs)


async def main() -> None:
    """Example usage"""
    load_dotenv()

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

    extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)

    try:
        await extractor.initialize()

        # Analyze repository - direct Neo4j, no LLM processing!
        repo_url = "https://github.com/getzep/graphiti.git"
        await extractor.analyze_repository(repo_url)

        # Direct graph queries
        print("\\n=== Direct Neo4j Queries ===")

        # Which files import from models?
        results = await extractor.search_graph("files_importing", target="models")
        if results:
            print(f"\\nFiles importing from 'models': {len(results)}")
            for result in results[:3]:
                print(f"- {result['file']} imports {result['imports']}")

        # What classes are in a specific file?
        results = await extractor.search_graph(
            "classes_in_file", file_path="pydantic_ai/models/openai.py",
        )
        if results:
            print(f"\\nClasses in openai.py: {len(results)}")
            for result in results:
                print(f"- {result['class_name']}")

        # What methods does OpenAIModel have?
        results = await extractor.search_graph(
            "methods_of_class", class_name="OpenAIModel",
        )
        if results:
            print(f"\\nMethods of OpenAIModel: {len(results)}")
            for result in results[:5]:
                print(f"- {result['method_name']}({', '.join(result['args'])})")

    finally:
        await extractor.close()


if __name__ == "__main__":
    asyncio.run(main())
