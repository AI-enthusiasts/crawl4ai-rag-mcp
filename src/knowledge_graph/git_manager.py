"""
Git Repository Manager for enhanced Git operations.

This module provides comprehensive Git repository management including
cloning, updating, branch/tag management, and history analysis.
"""

import asyncio
import logging
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.exceptions import GitError, RepositoryError

logger = logging.getLogger(__name__)


class GitRepositoryManager:
    """Manages Git repository operations with async support."""

    def __init__(self) -> None:
        """Initialize the Git repository manager."""
        self.logger = logger

    async def clone_repository(
        self,
        url: str,
        target_dir: str,
        branch: str | None = None,
        depth: int | None = None,
        single_branch: bool = False,
    ) -> str:
        """
        Clone a Git repository with advanced options.

        Args:
            url: Repository URL (GitHub, GitLab, local path, etc.)
            target_dir: Target directory for cloning
            branch: Specific branch to clone (default: main/master)
            depth: Clone depth for shallow cloning (default: full history)
            single_branch: Whether to clone only specified branch

        Returns:
            Path to the cloned repository

        Raises:
            RuntimeError: If cloning fails
        """
        self.logger.info("Cloning repository from %s to %s", url, target_dir)

        # Clean up existing directory if it exists
        if Path(target_dir).exists():
            self.logger.info("Removing existing directory: %s", target_dir)
            await self._remove_directory(target_dir)

        # Build git clone command
        cmd = ["git", "clone"]

        if depth:
            cmd.extend(["--depth", str(depth)])

        if single_branch:
            cmd.append("--single-branch")

        if branch:
            cmd.extend(["--branch", branch])

        cmd.extend([url, target_dir])

        # Execute clone command
        try:
            await self._run_git_command(cmd)
            self.logger.info("Repository cloned successfully to %s", target_dir)
            return target_dir
        except GitError:
            raise
        except Exception as e:
            msg = "Failed to clone repository: %s"
            self.logger.error(msg, e)
            raise GitError(msg % e) from e
    async def validate_repository_size(
        self,
        url: str,
        max_size_mb: int = 500,
        max_file_count: int = 10000,
        min_free_space_gb: float = 1.0,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate repository size before cloning to prevent resource exhaustion.

        Args:
            url: Repository URL
            max_size_mb: Maximum allowed repository size in MB
            max_file_count: Maximum allowed number of files
            min_free_space_gb: Minimum required free disk space in GB

        Returns:
            Tuple of (is_valid, info_dict) where info_dict contains:
                - estimated_size_mb: Estimated repository size
                - file_count: Number of files (if available)
                - free_space_gb: Available disk space
                - errors: List of validation errors
        """
        info: dict[str, Any] = {
            "estimated_size_mb": 0,
            "file_count": 0,
            "free_space_gb": 0,
            "errors": [],
        }

        try:
            # Check available disk space first
            disk_usage = shutil.disk_usage("/")
            info["free_space_gb"] = disk_usage.free / (1024**3)

            if info["free_space_gb"] < min_free_space_gb:
                error_msg = (
                    f"Insufficient disk space: "
                    f"{info['free_space_gb']:.2f}GB available, "
                    f"{min_free_space_gb:.2f}GB required"
                )
                info["errors"].append(error_msg)
                return False, info

            # Try to get repository size using git ls-remote
            self.logger.info("Validating repository size for %s", url)

            # Method 1: Try to get size estimate using shallow clone with depth 1
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Clone with minimal depth to check size
                    # Using --bare for more accurate size estimation
                    cmd = [
                        "git", "clone", "--bare", "--depth", "1",
                        url, temp_dir + "/test.git",
                    ]

                    await self._run_git_command(cmd)

                    # Get repository info from the shallow clone
                    repo_info = await self.get_repository_info(temp_dir + "/test.git")

                    # Parse size from repo info
                    if repo_info.get("size"):
                        size_str = repo_info["size"]
                        if "MiB" in size_str:
                            size_val = float(size_str.replace(" MiB", ""))
                            info["estimated_size_mb"] = size_val
                        elif "KiB" in size_str:
                            size_val = float(size_str.replace(" KiB", ""))
                            info["estimated_size_mb"] = size_val / 1024
                        elif "GiB" in size_str:
                            size_val = float(size_str.replace(" GiB", ""))
                            info["estimated_size_mb"] = size_val * 1024

                    info["file_count"] = repo_info.get("file_count", 0)

                except Exception as e:
                    self.logger.warning(
                        "Could not get exact size, trying alternative method: %s", e,
                    )

                    # Method 2: Use GitHub API if it's a GitHub repository
                    if "github.com" in url:
                        info = await self._check_github_api_size(url, info)

            # Validate against limits
            if info["estimated_size_mb"] > max_size_mb:
                info["errors"].append(
                    f"Repository too large: {info['estimated_size_mb']:.2f}MB exceeds "
                    f"limit of {max_size_mb}MB",
                )
                return False, info

            if info["file_count"] > max_file_count:
                error_msg = (
                    f"Too many files: {info['file_count']} "
                    f"exceeds limit of {max_file_count}"
                )
                info["errors"].append(error_msg)
                return False, info

            # Check if we have enough space for the repository (with 2x safety margin)
            required_space_gb = (info["estimated_size_mb"] * 2) / 1024
            if info["free_space_gb"] < required_space_gb:
                error_msg = (
                    f"Insufficient space for repository: "
                    f"{required_space_gb:.2f}GB needed, "
                    f"{info['free_space_gb']:.2f}GB available"
                )
                info["errors"].append(error_msg)
                return False, info

            self.logger.info(
                "Repository validation passed: %.2fMB, %s files",
                info["estimated_size_mb"], info["file_count"],
            )
            return True, info

        except Exception as e:
            self.logger.exception("Error validating repository")
            info["errors"].append(f"Validation error: {e!s}")
            return False, info

    async def _check_github_api_size(
        self, url: str, info: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Check repository size using GitHub API.

        Args:
            url: GitHub repository URL
            info: Existing info dictionary to update

        Returns:
            Updated info dictionary
        """
        try:
            # Extract owner and repo from URL
            import re
            match = re.search(r"github\.com[/:]([^/]+)/([^/.]+)", url)
            if not match:
                return info

            owner, repo = match.groups()
            repo = repo.replace(".git", "")

            # Use GitHub API to get repository info
            import json
            import urllib.request

            api_url = f"https://api.github.com/repos/{owner}/{repo}"

            try:
                with urllib.request.urlopen(api_url) as response:
                    data = json.loads(response.read())

                    # GitHub API returns size in KB
                    if "size" in data:
                        info["estimated_size_mb"] = data["size"] / 1024

                    # Note: file count is not available via GitHub API
                    self.logger.info(
                        "GitHub API reports repository size: %.2fMB",
                        info["estimated_size_mb"],
                    )
            except OSError as api_error:
                self.logger.debug("GitHub API network error: %s", api_error)
            except Exception:
                self.logger.exception("Unexpected error with GitHub API")

        except Exception:
            self.logger.exception("Unexpected error checking GitHub API")

        return info

    async def clone_repository_with_validation(
        self,
        url: str,
        target_dir: str,
        branch: str | None = None,
        depth: int | None = None,
        single_branch: bool = False,
        max_size_mb: int = 500,
        max_file_count: int = 10000,
        min_free_space_gb: float = 1.0,
        force: bool = False,
    ) -> str:
        """
        Clone a repository with size validation.

        Args:
            url: Repository URL
            target_dir: Target directory for cloning
            branch: Specific branch to clone
            depth: Clone depth for shallow cloning
            single_branch: Whether to clone only specified branch
            max_size_mb: Maximum allowed repository size in MB
            max_file_count: Maximum allowed number of files
            min_free_space_gb: Minimum required free disk space in GB
            force: Force clone even if validation fails (use with caution)

        Returns:
            Path to the cloned repository

        Raises:
            RuntimeError: If validation fails or cloning fails
        """
        # Validate repository size unless forced
        if not force:
            is_valid, info = await self.validate_repository_size(
                url, max_size_mb, max_file_count, min_free_space_gb,
            )

            if not is_valid:
                errors = "; ".join(info["errors"])
                msg = f"Repository validation failed: {errors}"
                self.logger.error(msg)
                raise RepositoryError(msg)

            self.logger.info(
                "Repository validation passed - "
                "Size: %.2fMB, Files: %s, Free space: %.2fGB",
                info["estimated_size_mb"],
                info["file_count"],
                info["free_space_gb"],
            )
        else:
            self.logger.warning("Skipping size validation (force=True)")

        # Proceed with cloning
        return await self.clone_repository(
            url=url,
            target_dir=target_dir,
            branch=branch,
            depth=depth,
            single_branch=single_branch,
        )


    async def update_repository(
        self, repo_dir: str, branch: str | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing repository (pull latest changes).

        Args:
            repo_dir: Path to the repository
            branch: Branch to update (default: current branch)

        Returns:
            Update information including changed files
        """
        self.logger.info("Updating repository at %s", repo_dir)

        # Store current commit
        old_commit = await self.get_current_commit(repo_dir)

        # Checkout branch if specified
        if branch:
            await self.checkout_branch(repo_dir, branch)

        # Pull latest changes
        cmd = ["git", "pull", "--ff-only"]
        await self._run_git_command(cmd, cwd=repo_dir)

        # Get new commit
        new_commit = await self.get_current_commit(repo_dir)

        # Get changed files
        changed_files = []
        if old_commit != new_commit:
            cmd = ["git", "diff", "--name-only", old_commit, new_commit]
            result = await self._run_git_command(cmd, cwd=repo_dir)
            changed_files = result.strip().split("\n") if result else []

        return {
            "old_commit": old_commit,
            "new_commit": new_commit,
            "changed_files": changed_files,
            "updated": old_commit != new_commit,
        }

    async def get_branches(self, repo_dir: str) -> list[dict[str, str]]:
        """
        Get all branches in the repository.

        Args:
            repo_dir: Path to the repository

        Returns:
            List of branch information
        """
        cmd = [
            "git",
            "branch",
            "-a",
            "--format=%(refname:short)|%(committerdate:iso)|%(subject)",
        ]
        result = await self._run_git_command(cmd, cwd=repo_dir)

        branches = []
        for line in result.strip().split("\n"):
            if line:
                parts = line.split("|", 2)
                if len(parts) >= 3:
                    branches.append(
                        {
                            "name": parts[0].replace("origin/", ""),
                            "last_commit_date": parts[1],
                            "last_commit_message": parts[2],
                        },
                    )

        return branches

    async def get_tags(self, repo_dir: str) -> list[dict[str, str]]:
        """
        Get all tags in the repository.

        Args:
            repo_dir: Path to the repository

        Returns:
            List of tag information
        """
        cmd = [
            "git",
            "tag",
            "-l",
            "--format=%(refname:short)|%(creatordate:iso)|%(subject)",
        ]
        result = await self._run_git_command(cmd, cwd=repo_dir)

        tags = []
        for line in result.strip().split("\n"):
            if line:
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    tags.append(
                        {
                            "name": parts[0],
                            "date": parts[1] if len(parts) > 1 else "",
                            "message": parts[2] if len(parts) > 2 else "",
                        },
                    )

        return tags

    async def get_commits(
        self,
        repo_dir: str,
        limit: int = 100,
        branch: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get commit history.

        Args:
            repo_dir: Path to the repository
            limit: Maximum number of commits to retrieve
            branch: Specific branch (default: current branch)

        Returns:
            List of commit information
        """
        cmd = [
            "git",
            "log",
            f"--max-count={limit}",
            "--pretty=format:%H|%an|%ae|%at|%s",
            "--no-merges",
        ]

        if branch:
            cmd.append(branch)

        result = await self._run_git_command(cmd, cwd=repo_dir)

        commits = []
        for line in result.strip().split("\n"):
            if line:
                parts = line.split("|", 4)
                if len(parts) >= 5:
                    commits.append(
                        {
                            "hash": parts[0],
                            "author_name": parts[1],
                            "author_email": parts[2],
                            "timestamp": int(parts[3]),
                            "date": datetime.fromtimestamp(int(parts[3])).isoformat(),
                            "message": parts[4],
                        },
                    )

        return commits

    async def get_file_history(
        self,
        repo_dir: str,
        file_path: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get commit history for a specific file.

        Args:
            repo_dir: Path to the repository
            file_path: Path to the file relative to repo root
            limit: Maximum number of commits

        Returns:
            List of commits that modified the file
        """
        cmd = [
            "git",
            "log",
            f"--max-count={limit}",
            "--pretty=format:%H|%an|%at|%s",
            "--follow",
            "--",
            file_path,
        ]

        result = await self._run_git_command(cmd, cwd=repo_dir)

        history = []
        for line in result.strip().split("\n"):
            if line:
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    history.append(
                        {
                            "hash": parts[0],
                            "author": parts[1],
                            "timestamp": int(parts[2]),
                            "date": datetime.fromtimestamp(int(parts[2])).isoformat(),
                            "message": parts[3],
                        },
                    )

        return history

    async def checkout_branch(self, repo_dir: str, branch: str) -> None:
        """
        Checkout a specific branch.

        Args:
            repo_dir: Path to the repository
            branch: Branch name to checkout
        """
        cmd = ["git", "checkout", branch]
        await self._run_git_command(cmd, cwd=repo_dir)
        self.logger.info("Checked out branch: %s", branch)

    async def get_current_commit(self, repo_dir: str) -> str:
        """
        Get the current commit hash.

        Args:
            repo_dir: Path to the repository

        Returns:
            Current commit hash
        """
        cmd = ["git", "rev-parse", "HEAD"]
        result = await self._run_git_command(cmd, cwd=repo_dir)
        return result.strip()

    async def get_repository_info(self, repo_dir: str) -> dict[str, Any]:
        """
        Get comprehensive repository information.

        Args:
            repo_dir: Path to the repository

        Returns:
            Repository metadata including size, file count, etc.
        """
        info: dict[str, Any] = {}

        # Get remote URL
        try:
            cmd = ["git", "config", "--get", "remote.origin.url"]
            info["remote_url"] = (
                await self._run_git_command(cmd, cwd=repo_dir)
            ).strip()
        except GitError as e:
            self.logger.debug("Failed to get remote URL: %s", e)
            info["remote_url"] = None
        except Exception:
            self.logger.exception("Unexpected error getting remote URL")
            info["remote_url"] = None

        # Get current branch
        try:
            cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            info["current_branch"] = (
                await self._run_git_command(cmd, cwd=repo_dir)
            ).strip()
        except GitError as e:
            self.logger.debug("Failed to get current branch: %s", e)
            info["current_branch"] = None
        except Exception:
            self.logger.exception("Unexpected error getting current branch")
            info["current_branch"] = None

        # Get file statistics
        try:
            cmd = ["git", "ls-files"]
            files = (await self._run_git_command(cmd, cwd=repo_dir)).strip().split("\n")
            info["file_count"] = len([f for f in files if f])

            # Count by extension
            extensions: dict[str, int] = {}
            for file in files:
                if file and "." in file:
                    ext = file.split(".")[-1].lower()
                    extensions[ext] = extensions.get(ext, 0) + 1
            info["file_extensions"] = extensions
        except GitError as e:
            self.logger.debug("Failed to get file statistics: %s", e)
            info["file_count"] = 0
            info["file_extensions"] = {}
        except Exception:
            self.logger.exception("Unexpected error getting file statistics")
            info["file_count"] = 0
            info["file_extensions"] = {}

        # Get repository size
        try:
            cmd = ["git", "count-objects", "-v", "-H"]
            result = await self._run_git_command(cmd, cwd=repo_dir)
            for line in result.strip().split("\n"):
                if "size-pack:" in line:
                    info["size"] = line.split(":")[1].strip()
        except GitError as e:
            self.logger.debug("Failed to get repository size: %s", e)
            info["size"] = "unknown"
        except Exception:
            self.logger.exception("Unexpected error getting repository size")
            info["size"] = "unknown"

        # Get contributor count
        try:
            cmd = ["git", "log", "--format=%an"]
            authors = (
                (await self._run_git_command(cmd, cwd=repo_dir)).strip().split("\n")
            )
            info["contributor_count"] = len({a for a in authors if a})
        except GitError as e:
            self.logger.debug("Failed to get contributor count: %s", e)
            info["contributor_count"] = 0
        except Exception:
            self.logger.exception("Unexpected error getting contributor count")
            info["contributor_count"] = 0

        return info

    async def is_git_repository(self, path: str) -> bool:
        """
        Check if a directory is a Git repository.

        Args:
            path: Directory path to check

        Returns:
            True if it's a Git repository
        """
        try:
            cmd = ["git", "rev-parse", "--git-dir"]
            await self._run_git_command(cmd, cwd=path)
            return True
        except GitError:
            return False
        except Exception:
            self.logger.exception(
                "Unexpected error checking if path is git repository",
            )
            return False

    async def get_changed_files(
        self,
        repo_dir: str,
        from_commit: str,
        to_commit: str = "HEAD",
    ) -> list[dict[str, Any]]:
        """
        Get files changed between two commits.

        Args:
            repo_dir: Path to the repository
            from_commit: Starting commit hash
            to_commit: Ending commit hash (default: HEAD)

        Returns:
            List of changed files with change type
        """
        cmd = ["git", "diff", "--name-status", from_commit, to_commit]
        result = await self._run_git_command(cmd, cwd=repo_dir)

        changed_files = []
        for line in result.strip().split("\n"):
            if line:
                parts = line.split("\t", 1)
                if len(parts) >= 2:
                    status_map = {
                        "A": "added",
                        "M": "modified",
                        "D": "deleted",
                        "R": "renamed",
                        "C": "copied",
                    }
                    changed_files.append(
                        {
                            "status": status_map.get(parts[0][0], "unknown"),
                            "file": parts[1],
                        },
                    )

        return changed_files

    async def _run_git_command(
        self,
        cmd: list[str],
        cwd: str | None = None,
    ) -> str:
        """
        Run a git command asynchronously.

        Args:
            cmd: Command arguments
            cwd: Working directory

        Returns:
            Command output

        Raises:
            RuntimeError: If command fails
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
                msg = f"Git command failed: {error_msg}"
                raise GitError(msg)

            return stdout.decode("utf-8")
        except GitError:
            raise
        except Exception as e:
            cmd_str = " ".join(cmd)
            self.logger.exception("Unexpected error running git command: %s", cmd_str)
            raise GitError("Git command execution failed: %s" % e) from e

    async def _remove_directory(self, path: str) -> None:
        """
        Remove a directory with proper error handling.

        Args:
            path: Directory path to remove
        """

        def handle_remove_readonly(
            func: Callable[[str], None], path: str, exc: Any,
        ) -> None:
            try:
                if Path(path).exists():
                    Path(path).chmod(0o777)
                    func(path)
            except PermissionError:
                self.logger.warning("Could not remove %s - file in use", path)

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: shutil.rmtree(path, onerror=handle_remove_readonly),
            )
        except Exception as e:
            self.logger.warning("Could not fully remove %s: %s", path, e)
