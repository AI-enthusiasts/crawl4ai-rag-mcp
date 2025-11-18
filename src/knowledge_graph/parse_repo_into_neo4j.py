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
        self.repo_min_free_space_gb = float(os.environ.get("REPO_MIN_FREE_SPACE_GB", "1.0"))
        self.repo_allow_size_override = os.environ.get("REPO_ALLOW_SIZE_OVERRIDE", "false").lower() == "true"

        logger.info("Git metadata collection enabled with GitRepositoryManager")
        logger.info(f"Multi-language support enabled for: {', '.join(self.analyzer_factory.get_supported_languages())}")
        logger.info(f"Repository limits - Max size: {self.repo_max_size_mb}MB, Max files: {self.repo_max_file_count}, "
                   f"Min free space: {self.repo_min_free_space_gb}GB, Allow override: {self.repo_allow_size_override}")

    async def initialize(self) -> None:
        """Initialize Neo4j connection"""
        logger.info("Initializing Neo4j connection...")

        # Import notification suppression (available in neo4j>=5.21.0)
        try:
            from neo4j import NotificationMinimumSeverity
            # Create Neo4j driver with notification suppression
            self.driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                warn_notification_severity=NotificationMinimumSeverity.OFF,
            )
        except (ImportError, AttributeError):
            # Fallback for older versions - use logging suppression
            import logging
            logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
            self.driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )

        # Clear existing data
        # logger.info("Clearing existing data...")
        # async with self.driver.session() as session:
        #     await session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j connection initialized successfully")

    async def clear_repository_data(self, repo_name: str) -> None:
        """Delegate to neo4j.clear_repository_data"""
        await clear_repository_data(self.driver, repo_name)

    async def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()

    async def validate_before_processing(self, repo_url: str) -> tuple[bool, dict[str, Any]]:
        """
        Validate repository before processing.

        Args:
            repo_url: Repository URL to validate

        Returns:
            Tuple of (is_valid, info) with validation details
        """
        if not self.git_manager:
            logger.warning("GitRepositoryManager not available, skipping validation")
            return True, {"warning": "Validation skipped - GitRepositoryManager not available"}

        try:
            is_valid, info = await self.git_manager.validate_repository_size(
                url=repo_url,
                max_size_mb=self.repo_max_size_mb,
                max_file_count=self.repo_max_file_count,
                min_free_space_gb=self.repo_min_free_space_gb,
            )

            if not is_valid and not self.repo_allow_size_override:
                logger.error(f"Repository validation failed: {info.get('errors', [])}")
                return False, info
            if not is_valid and self.repo_allow_size_override:
                logger.warning(f"Repository exceeds limits but override is enabled: {info.get('errors', [])}")
                info["override_applied"] = True
                return True, info

            return is_valid, info
        except RepositoryError as e:
            logger.error(f"Repository validation failed: {e}")
            return False, {"errors": [str(e)]}
        except Exception as e:
            logger.exception(f"Unexpected error during validation: {e}")
            return False, {"errors": [str(e)]}

    async def clone_repo(self, repo_url: str, target_dir: str, branch: str | None = None, force: bool = False) -> str:
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
        logger.info(f"Cloning repository to: {target_dir}")

        # Use GitRepositoryManager with validation if available
        if self.git_manager:
            try:
                # Use the new validation method
                return await self.git_manager.clone_repository_with_validation(
                    url=repo_url,
                    target_dir=target_dir,
                    branch=branch,
                    depth=1,  # Keep shallow clone for performance
                    single_branch=bool(branch),
                    max_size_mb=self.repo_max_size_mb,
                    max_file_count=self.repo_max_file_count,
                    min_free_space_gb=self.repo_min_free_space_gb,
                    force=force or self.repo_allow_size_override,
                )
            except RuntimeError:
                # Re-raise validation errors
                raise
            except GitError as e:
                logger.error(f"Git operation failed: {e}")
                raise
            except Exception as e:
                logger.warning(f"GitRepositoryManager failed, falling back to subprocess: {e}")

        # Fallback to original implementation
        if Path(target_dir).exists():
            logger.info(f"Removing existing directory: {target_dir}")
            try:
                def handle_remove_readonly(func: Callable[[str], None], path: str, exc: Any) -> None:
                    try:
                        if Path(path).exists():
                            Path(path).chmod(0o777)
                            func(path)
                    except PermissionError:
                        logger.warning(f"Could not remove {path} - file in use, skipping")
                shutil.rmtree(target_dir, onerror=handle_remove_readonly)
            except Exception as e:
                logger.warning(f"Could not fully remove {target_dir}: {e}. Proceeding anyway...")

        logger.info(f"Running git clone from {repo_url}")
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_url, target_dir])

        try:
            subprocess.run(cmd, check=True)
            logger.info("Repository cloned successfully")
            return target_dir
        except subprocess.CalledProcessError as e:
            msg = f"Git clone failed: {e}"
            logger.error(msg)
            raise GitError(msg) from e

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
                logger.info(f"Extracting Git metadata from {repo_dir}")

                # Get repository info
                metadata["info"] = await self.git_manager.get_repository_info(repo_dir)
                logger.debug(f"Repository info: {metadata['info']}")

                # Get branches
                metadata["branches"] = await self.git_manager.get_branches(repo_dir)
                logger.debug(f"Found {len(metadata['branches'])} branches")

                # Get tags
                metadata["tags"] = await self.git_manager.get_tags(repo_dir)
                logger.debug(f"Found {len(metadata['tags'])} tags")

                # Get recent commits (last 10)
                metadata["recent_commits"] = await self.git_manager.get_commits(repo_dir, limit=10)
                logger.debug(f"Found {len(metadata['recent_commits'])} recent commits")

                logger.info(f"Successfully extracted Git metadata: {len(metadata['branches'])} branches, "
                          f"{len(metadata['tags'])} tags, {len(metadata['recent_commits'])} commits")
            except GitError as e:
                logger.error(f"Git operation failed during metadata extraction: {e}")
                logger.warning("Continuing without Git metadata")
            except Exception as e:
                logger.exception(f"Unexpected error extracting Git metadata: {e}")
                logger.warning("Continuing without Git metadata")
        else:
            logger.warning("GitRepositoryManager not available - skipping Git metadata extraction")

        return metadata

    def get_python_files(self, repo_path: str) -> list[Path]:
        """Get Python files, focusing on main source directories"""
        python_files = []
        exclude_dirs = {
            "tests", "test", "__pycache__", ".git", "venv", "env",
            "node_modules", "build", "dist", ".pytest_cache", "docs",
            "examples", "example", "demo", "benchmark",
        }

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

            for file in files:
                if file.endswith(".py") and not file.startswith("test_"):
                    file_path = Path(root) / file
                    if (file_path.stat().st_size < 500_000 and
                        file not in ["setup.py", "conftest.py"]):
                        python_files.append(file_path)

        return python_files

    def get_code_files(self, repo_path: str) -> dict[str, list[Path]]:
        """Get all supported code files, organized by language"""
        code_files: dict[str, list[Path]] = {
            "python": [],
            "javascript": [],
            "typescript": [],
            "go": [],
        }

        exclude_dirs = {
            "tests", "test", "__pycache__", ".git", "venv", "env",
            "node_modules", "build", "dist", ".pytest_cache", "docs",
            "examples", "example", "demo", "benchmark", "vendor",
            ".next", ".nuxt", "coverage", "lib", "out",
        }

        supported_extensions = self.analyzer_factory.get_supported_extensions()

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                # Skip test files and large files
                if (ext in supported_extensions and
                    not file.startswith("test_") and
                    not file.endswith(".test" + ext) and
                    not file.endswith(".spec" + ext) and
                    file_path.stat().st_size < 500_000):

                    # Categorize by language
                    if ext == ".py":
                        if not any(skip in str(file_path) for skip in ["migrations/", "pb2.py", "_pb2_grpc.py"]):
                            code_files["python"].append(file_path)
                    elif ext in [".js", ".jsx", ".mjs", ".cjs"]:
                        if not any(skip in str(file_path) for skip in [".min.js", ".bundle.js", "webpack"]):
                            code_files["javascript"].append(file_path)
                    elif ext in [".ts", ".tsx"]:
                        if not any(skip in str(file_path) for skip in [".d.ts", ".min.js"]):
                            code_files["typescript"].append(file_path)
                    elif ext == ".go":
                        if not any(skip in str(file_path) for skip in ["_test.go", ".pb.go"]):
                            code_files["go"].append(file_path)

        return code_files

    async def analyze_repository(self, repo_url: str, temp_dir: str | None = None, branch: str | None = None, force: bool = False) -> None:
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
        logger.info(f"Analyzing repository: {repo_name}")

        # Clear existing data for this repository before re-processing
        await self.clear_repository_data(repo_name)

        # Set default temp_dir to repos folder at script level
        if temp_dir is None:
            script_dir = Path(__file__).parent
            temp_dir = str(script_dir / "repos" / repo_name)

        # Clone and analyze
        repo_path = Path(await self.clone_repo(repo_url, temp_dir, branch, force=force))

        try:
            # Get all code files, organized by language
            logger.info("Getting code files for all supported languages...")
            code_files = self.get_code_files(str(repo_path))

            total_files = sum(len(files) for files in code_files.values())
            logger.info(f"Found {total_files} code files to analyze:")
            for lang, files in code_files.items():
                if files:
                    logger.info(f"  - {lang}: {len(files)} files")

            # First pass: identify project modules (for Python)
            logger.info("Identifying project modules...")
            project_modules = set()
            for file_path in code_files.get("python", []):
                relative_path = str(file_path.relative_to(repo_path))
                module_parts = relative_path.replace("/", ".").replace(".py", "").split(".")
                if len(module_parts) > 0 and not module_parts[0].startswith("."):
                    project_modules.add(module_parts[0])

            if project_modules:
                logger.info(f"Identified Python project modules: {sorted(project_modules)}")

            # Second pass: analyze files and collect data
            modules_data = []
            file_counter = 0

            # Analyze Python files
            for file_path in code_files.get("python", []):
                if file_counter % 20 == 0:
                    logger.info(f"Analyzing file {file_counter+1}/{total_files}: {file_path.name}")
                file_counter += 1

                analysis = self.analyzer.analyze_python_file(file_path, repo_path, project_modules)
                if analysis:
                    analysis["language"] = "Python"
                    modules_data.append(analysis)

            # Analyze JavaScript/TypeScript files
            js_analyzer = self.analyzer_factory.get_analyzer(".js")
            for lang, _ext in [("javascript", ".js"), ("typescript", ".ts")]:
                for file_path in code_files.get(lang, []):
                    if file_counter % 20 == 0:
                        logger.info(f"Analyzing file {file_counter+1}/{total_files}: {file_path.name}")
                    file_counter += 1

                    if js_analyzer:
                        analysis = await js_analyzer.analyze_file(str(file_path), str(repo_path))
                        if analysis:
                            modules_data.append(analysis)

            # Analyze Go files
            go_analyzer = self.analyzer_factory.get_analyzer(".go")
            for file_path in code_files.get("go", []):
                if file_counter % 20 == 0:
                    logger.info(f"Analyzing file {file_counter+1}/{total_files}: {file_path.name}")
                file_counter += 1

                if go_analyzer:
                    analysis = await go_analyzer.analyze_file(str(file_path), str(repo_path))
                    if analysis:
                        modules_data.append(analysis)

            logger.info(f"Found {len(modules_data)} files with content")

            # Get Git metadata if available
            git_metadata = await self.get_repository_metadata(str(repo_path))

            # Create nodes and relationships in Neo4j
            logger.info("Creating nodes and relationships in Neo4j...")
            await self._create_graph(repo_name, modules_data, git_metadata)

            # Print summary
            total_classes = sum(len(mod["classes"]) for mod in modules_data)
            total_methods = sum(len(cls["methods"]) for mod in modules_data for cls in mod["classes"])
            total_functions = sum(len(mod["functions"]) for mod in modules_data)
            total_imports = sum(len(mod["imports"]) for mod in modules_data)

            logger.info(f"\n=== Direct Neo4j Repository Analysis for {repo_name} ===")
            logger.info(f"Files processed: {len(modules_data)}")
            logger.info(f"Classes created: {total_classes}")
            logger.info(f"Methods created: {total_methods}")
            logger.info(f"Functions created: {total_functions}")
            logger.info(f"Import relationships: {total_imports}")

            logger.info(f"Successfully created Neo4j graph for {repo_name}")

        finally:
            if Path(temp_dir).exists():
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                try:
                    def handle_remove_readonly(func: Callable[[str], None], path: str, exc: Any) -> None:
                        try:
                            if Path(path).exists():
                                Path(path).chmod(0o777)
                                func(path)
                        except PermissionError:
                            logger.warning(f"Could not remove {path} - file in use, skipping")

                    shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
                    logger.info("Cleanup completed")
                except Exception as e:
                    logger.warning(f"Cleanup failed: {e}. Directory may remain at {temp_dir}")
                    # Don't fail the whole process due to cleanup issues
                except GitError as e:
                    logger.error(f"Git-related cleanup failed: {e}")


    async def analyze_local_repository(self, local_path: str, repo_name: str) -> None:
        """
        Analyze a local Git repository without cloning.

        Args:
            local_path: Absolute path to the local repository
            repo_name: Repository name for Neo4j storage
        """
        from pathlib import Path

        logger.info(f"Analyzing local repository: {repo_name} at {local_path}")

        # Clear existing data for this repository before re-processing
        await self.clear_repository_data(repo_name)

        repo_path = Path(local_path)

        try:
            # Get all code files, organized by language
            logger.info("Getting code files for all supported languages...")
            code_files = self.get_code_files(str(repo_path))

            total_files = sum(len(files) for files in code_files.values())
            logger.info(f"Found {total_files} code files to analyze:")
            for lang, files in code_files.items():
                if files:
                    logger.info(f"  - {lang}: {len(files)} files")

            # First pass: identify project modules (for Python)
            logger.info("Identifying project modules...")
            project_modules = set()
            for file_path in code_files.get("python", []):
                module_name = self.analyzer._get_importable_module_name(
                    Path(file_path), repo_path, str(Path(file_path).relative_to(repo_path)),
                )
                if module_name and not module_name.startswith("test") and "__pycache__" not in module_name:
                    project_modules.add(module_name.split(".")[0])

            logger.info(f"Identified {len(project_modules)} project modules: {sorted(project_modules)}")

            # Second pass: analyze all files
            logger.info("Analyzing code structure...")
            all_modules = []

            for lang, files in code_files.items():
                if not files:
                    continue

                logger.info(f"Analyzing {len(files)} {lang} files...")
                analyzer = self.analyzer_factory.get_analyzer(lang)

                for file_path in files:
                    try:
                        file_path_obj = Path(file_path)
                        logger.debug(f"Processing {lang} file: {file_path_obj.relative_to(repo_path)}")

                        # Use language-specific analyzer
                        if lang == "python":
                            # Use Python analyzer with project modules context
                            module_data = self.analyzer.analyze_python_file(
                                file_path_obj, repo_path, project_modules,
                            )
                        else:
                            # Use appropriate analyzer for other languages
                            module_data = await analyzer.analyze_file(
                                str(file_path_obj), str(repo_path),
                            ) if analyzer else None

                        if module_data:
                            # Add language information
                            module_data["language"] = lang
                            all_modules.append(module_data)

                    except Exception as e:
                        logger.warning(f"Failed to analyze {file_path}: {e}")
                        continue

            logger.info(f"Successfully analyzed {len(all_modules)} files")

            # Get Git metadata for local repository
            git_metadata = {}
            if self.git_manager:
                try:
                    logger.info("Collecting Git metadata...")

                    # Get repository info
                    repo_info = await self.git_manager.get_repository_info(local_path)

                    # Get branches
                    branches = await self.git_manager.get_branches(local_path)

                    # Get recent commits
                    commits = await self.git_manager.get_commits(local_path, limit=50)

                    git_metadata = {
                        "info": repo_info,
                        "branches": branches,
                        "commits": commits,
                    }

                    logger.info(f"Collected metadata: {len(branches)} branches, {len(commits)} commits")

                except GitError as e:
                    logger.error(f"Git operation failed during metadata collection: {e}")
                except Exception as e:
                    logger.exception(f"Unexpected error collecting Git metadata: {e}")

            # Create Neo4j graph
            logger.info("Creating Neo4j graph...")
            await self._create_graph(repo_name, all_modules, git_metadata)

            logger.info(f"Analysis complete for local repository: {repo_name}")

        except (GitError, ParsingError, AnalysisError, RepositoryError) as e:
            logger.error(f"Analysis failed for local repository {local_path}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error analyzing local repository {local_path}: {e}")
            raise
    async def _create_graph(self, repo_name: str, modules_data: list[dict[str, Any]], git_metadata: dict[str, Any] | None = None) -> None:
        """Delegate to neo4j.create_graph"""
        await create_graph(self.driver, repo_name, modules_data, git_metadata)

    async def search_graph(self, query_type: str, **kwargs: Any) -> list[dict[str, Any]] | None:
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
        # repo_url = "https://github.com/pydantic/pydantic-ai.git"
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
        results = await extractor.search_graph("classes_in_file", file_path="pydantic_ai/models/openai.py")
        if results:
            print(f"\\nClasses in openai.py: {len(results)}")
            for result in results:
                print(f"- {result['class_name']}")

        # What methods does OpenAIModel have?
        results = await extractor.search_graph("methods_of_class", class_name="OpenAIModel")
        if results:
            print(f"\\nMethods of OpenAIModel: {len(results)}")
            for result in results[:5]:
                print(f"- {result['method_name']}({', '.join(result['args'])})")

    finally:
        await extractor.close()


if __name__ == "__main__":
    asyncio.run(main())
