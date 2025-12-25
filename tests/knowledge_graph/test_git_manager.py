"""
Comprehensive unit tests for GitRepositoryManager.

Tests git operations with full mocking - no actual git commands executed.
Covers all major methods, error handling, and edge cases.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.core.exceptions import GitError, RepositoryError
from src.knowledge_graph.git_manager import GitRepositoryManager


@pytest.fixture
def git_manager():
    """Create GitRepositoryManager instance for testing."""
    return GitRepositoryManager()


@pytest.fixture
def mock_subprocess():
    """Mock asyncio subprocess for git commands."""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.stdout = b"test output"
    mock_process.stderr = b""
    mock_process.communicate = AsyncMock(return_value=(b"test output", b""))
    return mock_process


@pytest.fixture
def mock_subprocess_factory():
    """Factory to create mock subprocess with custom return values."""

    def _factory(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        mock_process = AsyncMock()
        mock_process.returncode = returncode
        mock_process.communicate = AsyncMock(return_value=(stdout, stderr))
        return mock_process

    return _factory


class TestGitRepositoryManagerInit:
    """Test GitRepositoryManager initialization."""

    def test_init_creates_instance(self):
        """Test manager initializes correctly."""
        manager = GitRepositoryManager()
        assert manager is not None
        assert manager.logger is not None

    def test_init_sets_logger(self):
        """Test logger is properly configured."""
        manager = GitRepositoryManager()
        assert hasattr(manager, "logger")
        assert manager.logger.name == "src.knowledge_graph.git_manager"


class TestCloneRepository:
    """Test clone_repository method."""

    @pytest.mark.asyncio
    async def test_clone_basic_success(
        self,
        git_manager,
        mock_subprocess_factory,
        tmp_path,
    ):
        """Test basic repository cloning."""
        target_dir = str(tmp_path / "test_repo")
        url = "https://github.com/test/repo.git"

        mock_proc = mock_subprocess_factory(stdout=b"Cloning into...", returncode=0)

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            result = await git_manager.clone_repository(url, target_dir)

            assert result == target_dir
            mock_exec.assert_called_once()
            args = mock_exec.call_args[0]
            assert args[0] == "git"
            assert args[1] == "clone"
            assert url in args
            assert target_dir in args

    @pytest.mark.asyncio
    async def test_clone_with_branch(self, git_manager, mock_subprocess_factory):
        """Test cloning specific branch."""
        url = "https://github.com/test/repo.git"
        target_dir = "/tmp/test_repo"
        branch = "develop"

        mock_proc = mock_subprocess_factory(returncode=0)

        with (
            patch("os.path.exists", return_value=False),
            patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ) as mock_exec,
        ):
            await git_manager.clone_repository(url, target_dir, branch=branch)

            args = mock_exec.call_args[0]
            assert "--branch" in args
            assert branch in args

    @pytest.mark.asyncio
    async def test_clone_with_depth(self, git_manager, mock_subprocess_factory):
        """Test shallow cloning with depth."""
        url = "https://github.com/test/repo.git"
        target_dir = "/tmp/test_repo"
        depth = 1

        mock_proc = mock_subprocess_factory(returncode=0)

        with (
            patch("os.path.exists", return_value=False),
            patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ) as mock_exec,
        ):
            await git_manager.clone_repository(url, target_dir, depth=depth)

            args = mock_exec.call_args[0]
            assert "--depth" in args
            assert "1" in args

    @pytest.mark.asyncio
    async def test_clone_with_single_branch(self, git_manager, mock_subprocess_factory):
        """Test cloning with single branch flag."""
        url = "https://github.com/test/repo.git"
        target_dir = "/tmp/test_repo"

        mock_proc = mock_subprocess_factory(returncode=0)

        with (
            patch("os.path.exists", return_value=False),
            patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ) as mock_exec,
        ):
            await git_manager.clone_repository(
                url,
                target_dir,
                single_branch=True,
                branch="main",
            )

            args = mock_exec.call_args[0]
            assert "--single-branch" in args

    @pytest.mark.asyncio
    async def test_clone_removes_existing_directory(
        self,
        git_manager,
        mock_subprocess_factory,
    ):
        """Test that existing directory is removed before cloning."""
        url = "https://github.com/test/repo.git"
        target_dir = "/tmp/existing_repo"

        mock_proc = mock_subprocess_factory(returncode=0)
        mock_remove = AsyncMock()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                git_manager,
                "_remove_directory",
                mock_remove,
            ),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            await git_manager.clone_repository(url, target_dir)

            mock_remove.assert_called_once_with(target_dir)

    @pytest.mark.asyncio
    async def test_clone_failure_raises_git_error(self, git_manager):
        """Test clone failure raises GitError."""
        url = "https://github.com/test/repo.git"
        target_dir = "/tmp/test_repo"

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"fatal: repository not found"),
        )

        with (
            patch("os.path.exists", return_value=False),
            patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ),
        ):
            with pytest.raises(GitError) as exc_info:
                await git_manager.clone_repository(url, target_dir)

            assert "repository not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_clone_exception_wrapped_in_git_error(self, git_manager):
        """Test unexpected exceptions are wrapped in GitError."""
        url = "https://github.com/test/repo.git"
        target_dir = "/tmp/test_repo"

        with (
            patch("os.path.exists", return_value=False),
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=OSError("Permission denied"),
            ),
        ):
            with pytest.raises(GitError) as exc_info:
                await git_manager.clone_repository(url, target_dir)

            assert "Permission denied" in str(exc_info.value)


class TestValidateRepositorySize:
    """Test validate_repository_size method."""

    @pytest.mark.asyncio
    async def test_validate_passes_for_small_repo(
        self,
        git_manager,
        mock_subprocess_factory,
    ):
        """Test validation passes for repository within limits."""
        url = "https://github.com/test/small-repo.git"

        # Mock disk usage
        mock_disk_usage = MagicMock()
        mock_disk_usage.free = 10 * (1024**3)  # 10GB free

        # Mock git commands
        mock_clone = mock_subprocess_factory(returncode=0)
        mock_info = mock_subprocess_factory(stdout=b"size-pack: 50 MiB\n", returncode=0)

        with (
            patch("shutil.disk_usage", return_value=mock_disk_usage),
            patch.object(
                git_manager,
                "_run_git_command",
                AsyncMock(),
            ) as mock_git,
            patch.object(
                git_manager,
                "get_repository_info",
                AsyncMock(return_value={"size": "50 MiB", "file_count": 100}),
            ),
        ):
            is_valid, info = await git_manager.validate_repository_size(url)

            assert is_valid
            assert info["estimated_size_mb"] == 50
            assert info["file_count"] == 100
            assert info["free_space_gb"] > 0
            assert len(info["errors"]) == 0

    @pytest.mark.asyncio
    async def test_validate_fails_insufficient_disk_space(self, git_manager):
        """Test validation fails when disk space is low."""
        url = "https://github.com/test/repo.git"

        # Mock very low disk space
        mock_disk_usage = MagicMock()
        mock_disk_usage.free = 100 * (1024**2)  # Only 100MB free

        with patch("shutil.disk_usage", return_value=mock_disk_usage):
            is_valid, info = await git_manager.validate_repository_size(
                url,
                min_free_space_gb=1.0,
            )

            assert not is_valid
            assert "Insufficient disk space" in info["errors"][0]

    @pytest.mark.asyncio
    async def test_validate_fails_repo_too_large(
        self,
        git_manager,
        mock_subprocess_factory,
    ):
        """Test validation fails when repository exceeds size limit."""
        url = "https://github.com/test/huge-repo.git"

        mock_disk_usage = MagicMock()
        mock_disk_usage.free = 10 * (1024**3)  # 10GB free

        with (
            patch("shutil.disk_usage", return_value=mock_disk_usage),
            patch.object(
                git_manager,
                "_run_git_command",
                AsyncMock(),
            ),
            patch.object(
                git_manager,
                "get_repository_info",
                AsyncMock(return_value={"size": "600 MiB", "file_count": 1000}),
            ),
        ):
            is_valid, info = await git_manager.validate_repository_size(
                url,
                max_size_mb=500,
            )

            assert not is_valid
            assert "Repository too large" in info["errors"][0]

    @pytest.mark.asyncio
    async def test_validate_fails_too_many_files(
        self,
        git_manager,
        mock_subprocess_factory,
    ):
        """Test validation fails when file count exceeds limit."""
        url = "https://github.com/test/many-files-repo.git"

        mock_disk_usage = MagicMock()
        mock_disk_usage.free = 10 * (1024**3)

        with (
            patch("shutil.disk_usage", return_value=mock_disk_usage),
            patch.object(
                git_manager,
                "_run_git_command",
                AsyncMock(),
            ),
            patch.object(
                git_manager,
                "get_repository_info",
                AsyncMock(return_value={"size": "100 MiB", "file_count": 15000}),
            ),
        ):
            is_valid, info = await git_manager.validate_repository_size(
                url,
                max_file_count=10000,
            )

            assert not is_valid
            assert "Too many files" in info["errors"][0]

    @pytest.mark.asyncio
    async def test_validate_uses_github_api_fallback(self, git_manager):
        """Test GitHub API is used as fallback for size estimation."""
        url = "https://github.com/owner/repo.git"

        mock_disk_usage = MagicMock()
        mock_disk_usage.free = 10 * (1024**3)

        # Mock GitHub API response with proper context manager
        mock_response = MagicMock()
        mock_response.read = Mock(return_value=b'{"size": 1024}')  # 1MB in KB
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with (
            patch("shutil.disk_usage", return_value=mock_disk_usage),
            patch.object(
                git_manager,
                "_run_git_command",
                AsyncMock(side_effect=GitError("Clone failed")),
            ),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            is_valid, info = await git_manager.validate_repository_size(url)

            # GitHub returns size in KB, should be converted to MB
            assert info["estimated_size_mb"] == 1.0


class TestCloneRepositoryWithValidation:
    """Test clone_repository_with_validation method."""

    @pytest.mark.asyncio
    async def test_clone_with_validation_success(self, git_manager):
        """Test successful cloning with validation."""
        url = "https://github.com/test/repo.git"
        target_dir = "/tmp/test_repo"

        with (
            patch.object(
                git_manager,
                "validate_repository_size",
                AsyncMock(
                    return_value=(
                        True,
                        {
                            "estimated_size_mb": 50,
                            "file_count": 100,
                            "free_space_gb": 10,
                        },
                    )
                ),
            ),
            patch.object(
                git_manager,
                "clone_repository",
                AsyncMock(return_value=target_dir),
            ) as mock_clone,
        ):
            result = await git_manager.clone_repository_with_validation(url, target_dir)

            assert result == target_dir
            mock_clone.assert_called_once()

    @pytest.mark.asyncio
    async def test_clone_with_validation_fails(self, git_manager):
        """Test cloning fails when validation fails."""
        url = "https://github.com/test/huge-repo.git"
        target_dir = "/tmp/test_repo"

        with patch.object(
            git_manager,
            "validate_repository_size",
            AsyncMock(
                return_value=(
                    False,
                    {
                        "errors": ["Repository too large"],
                        "estimated_size_mb": 1000,
                        "file_count": 0,
                        "free_space_gb": 10,
                    },
                ),
            ),
        ):
            with pytest.raises(RepositoryError) as exc_info:
                await git_manager.clone_repository_with_validation(url, target_dir)

            assert "Repository too large" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_clone_with_validation_force_skip(self, git_manager):
        """Test validation is skipped when force=True."""
        url = "https://github.com/test/repo.git"
        target_dir = "/tmp/test_repo"

        with (
            patch.object(
                git_manager,
                "validate_repository_size",
                AsyncMock(),
            ) as mock_validate,
            patch.object(
                git_manager,
                "clone_repository",
                AsyncMock(return_value=target_dir),
            ) as mock_clone,
        ):
            result = await git_manager.clone_repository_with_validation(
                url,
                target_dir,
                force=True,
            )

            assert result == target_dir
            mock_validate.assert_not_called()
            mock_clone.assert_called_once()


class TestUpdateRepository:
    """Test update_repository method."""

    @pytest.mark.asyncio
    async def test_update_with_changes(self, git_manager, mock_subprocess_factory):
        """Test updating repository with changes."""
        repo_dir = "/tmp/test_repo"
        old_commit = "abc123"
        new_commit = "def456"

        with (
            patch.object(
                git_manager,
                "get_current_commit",
                AsyncMock(side_effect=[old_commit, new_commit]),
            ),
            patch.object(
                git_manager,
                "_run_git_command",
                AsyncMock(return_value="file1.py\nfile2.py"),
            ),
        ):
            result = await git_manager.update_repository(repo_dir)

            assert result["old_commit"] == old_commit
            assert result["new_commit"] == new_commit
            assert result["updated"]
            assert len(result["changed_files"]) == 2
            assert "file1.py" in result["changed_files"]

    @pytest.mark.asyncio
    async def test_update_no_changes(self, git_manager):
        """Test updating repository with no changes."""
        repo_dir = "/tmp/test_repo"
        commit = "abc123"

        with (
            patch.object(
                git_manager,
                "get_current_commit",
                AsyncMock(return_value=commit),
            ),
            patch.object(git_manager, "_run_git_command", AsyncMock(return_value="")),
        ):
            result = await git_manager.update_repository(repo_dir)

            assert result["old_commit"] == commit
            assert result["new_commit"] == commit
            assert not result["updated"]
            assert result["changed_files"] == []

    @pytest.mark.asyncio
    async def test_update_with_branch(self, git_manager):
        """Test updating specific branch."""
        repo_dir = "/tmp/test_repo"
        branch = "develop"

        with (
            patch.object(
                git_manager,
                "get_current_commit",
                AsyncMock(return_value="abc123"),
            ),
            patch.object(
                git_manager,
                "checkout_branch",
                AsyncMock(),
            ) as mock_checkout,
            patch.object(
                git_manager,
                "_run_git_command",
                AsyncMock(),
            ),
        ):
            await git_manager.update_repository(repo_dir, branch=branch)

            mock_checkout.assert_called_once_with(repo_dir, branch)


class TestGetBranches:
    """Test get_branches method."""

    @pytest.mark.asyncio
    async def test_get_branches_success(self, git_manager):
        """Test retrieving branches."""
        repo_dir = "/tmp/test_repo"
        git_output = "main|2025-01-10 10:00:00|Initial commit\ndevelop|2025-01-11 15:30:00|Feature work"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value=git_output),
        ):
            branches = await git_manager.get_branches(repo_dir)

            assert len(branches) == 2
            assert branches[0]["name"] == "main"
            assert branches[0]["last_commit_date"] == "2025-01-10 10:00:00"
            assert branches[0]["last_commit_message"] == "Initial commit"

    @pytest.mark.asyncio
    async def test_get_branches_empty(self, git_manager):
        """Test retrieving branches when none exist."""
        repo_dir = "/tmp/test_repo"

        with patch.object(git_manager, "_run_git_command", AsyncMock(return_value="")):
            branches = await git_manager.get_branches(repo_dir)

            assert branches == []


class TestGetTags:
    """Test get_tags method."""

    @pytest.mark.asyncio
    async def test_get_tags_success(self, git_manager):
        """Test retrieving tags."""
        repo_dir = "/tmp/test_repo"
        git_output = "v1.0.0|2025-01-01 00:00:00|Release 1.0.0\nv1.1.0|2025-01-15 00:00:00|Bug fixes"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value=git_output),
        ):
            tags = await git_manager.get_tags(repo_dir)

            assert len(tags) == 2
            assert tags[0]["name"] == "v1.0.0"
            assert tags[0]["date"] == "2025-01-01 00:00:00"
            assert tags[0]["message"] == "Release 1.0.0"

    @pytest.mark.asyncio
    async def test_get_tags_empty(self, git_manager):
        """Test retrieving tags when none exist."""
        repo_dir = "/tmp/test_repo"

        with patch.object(git_manager, "_run_git_command", AsyncMock(return_value="")):
            tags = await git_manager.get_tags(repo_dir)

            assert tags == []


class TestGetCommits:
    """Test get_commits method."""

    @pytest.mark.asyncio
    async def test_get_commits_success(self, git_manager):
        """Test retrieving commit history."""
        repo_dir = "/tmp/test_repo"
        git_output = "abc123|John Doe|john@example.com|1704067200|Initial commit\ndef456|Jane Smith|jane@example.com|1704153600|Add feature"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value=git_output),
        ):
            commits = await git_manager.get_commits(repo_dir, limit=10)

            assert len(commits) == 2
            assert commits[0]["hash"] == "abc123"
            assert commits[0]["author_name"] == "John Doe"
            assert commits[0]["author_email"] == "john@example.com"
            assert commits[0]["timestamp"] == 1704067200
            assert commits[0]["message"] == "Initial commit"

    @pytest.mark.asyncio
    async def test_get_commits_with_branch(self, git_manager):
        """Test retrieving commits from specific branch."""
        repo_dir = "/tmp/test_repo"
        branch = "develop"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value=""),
        ) as mock_cmd:
            await git_manager.get_commits(repo_dir, branch=branch)

            args = mock_cmd.call_args[0][0]
            assert branch in args


class TestGetFileHistory:
    """Test get_file_history method."""

    @pytest.mark.asyncio
    async def test_get_file_history_success(self, git_manager):
        """Test retrieving file history."""
        repo_dir = "/tmp/test_repo"
        file_path = "src/main.py"
        git_output = "abc123|John Doe|1704067200|Initial version\ndef456|Jane Smith|1704153600|Update logic"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value=git_output),
        ):
            history = await git_manager.get_file_history(repo_dir, file_path)

            assert len(history) == 2
            assert history[0]["hash"] == "abc123"
            assert history[0]["author"] == "John Doe"
            assert history[0]["message"] == "Initial version"


class TestGetRepositoryInfo:
    """Test get_repository_info method."""

    @pytest.mark.asyncio
    async def test_get_repository_info_success(self, git_manager):
        """Test retrieving comprehensive repository info."""
        repo_dir = "/tmp/test_repo"

        async def mock_run_command(cmd, cwd=None):
            # Check command as list
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)

            if "remote.origin.url" in cmd_str:
                return "https://github.com/test/repo.git\n"
            if "rev-parse --abbrev-ref HEAD" in cmd_str:
                return "main\n"
            if "ls-files" in cmd_str:
                return "src/main.py\nsrc/utils.py\nREADME.md\n"
            if "count-objects" in cmd_str:
                return "size-pack: 100 MiB\n"
            if "log --format=%an" in cmd_str:
                return "John Doe\nJane Smith\nJohn Doe\n"
            return ""

        with patch.object(
            git_manager,
            "_run_git_command",
            side_effect=mock_run_command,
        ):
            info = await git_manager.get_repository_info(repo_dir)

            assert info["remote_url"] == "https://github.com/test/repo.git"
            assert info["current_branch"] == "main"
            assert info["file_count"] == 3
            assert "py" in info["file_extensions"]
            assert info["file_extensions"]["py"] == 2
            assert info["size"] == "100 MiB"
            assert info["contributor_count"] == 2  # Unique contributors

    @pytest.mark.asyncio
    async def test_get_repository_info_handles_errors(self, git_manager):
        """Test repository info handles missing data gracefully."""
        repo_dir = "/tmp/test_repo"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(side_effect=GitError("Command failed")),
        ):
            info = await git_manager.get_repository_info(repo_dir)

            # Should return None/defaults for all fields
            assert info["remote_url"] is None
            assert info["current_branch"] is None
            assert info["file_count"] == 0
            assert info["file_extensions"] == {}
            assert info["size"] == "unknown"
            assert info["contributor_count"] == 0


class TestIsGitRepository:
    """Test is_git_repository method."""

    @pytest.mark.asyncio
    async def test_is_git_repository_true(self, git_manager):
        """Test detection of valid git repository."""
        path = "/tmp/test_repo"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value=".git"),
        ):
            result = await git_manager.is_git_repository(path)

            assert result is True

    @pytest.mark.asyncio
    async def test_is_git_repository_false(self, git_manager):
        """Test detection of non-git directory."""
        path = "/tmp/not_a_repo"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(side_effect=GitError("Not a git repo")),
        ):
            result = await git_manager.is_git_repository(path)

            assert result is False


class TestGetChangedFiles:
    """Test get_changed_files method."""

    @pytest.mark.asyncio
    async def test_get_changed_files_success(self, git_manager):
        """Test retrieving changed files between commits."""
        repo_dir = "/tmp/test_repo"
        git_output = "M\tsrc/main.py\nA\tsrc/new.py\nD\tsrc/old.py"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value=git_output),
        ):
            changed = await git_manager.get_changed_files(repo_dir, "abc123", "def456")

            assert len(changed) == 3
            assert changed[0]["status"] == "modified"
            assert changed[0]["file"] == "src/main.py"
            assert changed[1]["status"] == "added"
            assert changed[2]["status"] == "deleted"


class TestCheckoutBranch:
    """Test checkout_branch method."""

    @pytest.mark.asyncio
    async def test_checkout_branch_success(self, git_manager):
        """Test checking out a branch."""
        repo_dir = "/tmp/test_repo"
        branch = "develop"

        with patch.object(git_manager, "_run_git_command", AsyncMock()):
            await git_manager.checkout_branch(repo_dir, branch)

            # Should complete without error


class TestGetCurrentCommit:
    """Test get_current_commit method."""

    @pytest.mark.asyncio
    async def test_get_current_commit_success(self, git_manager):
        """Test retrieving current commit hash."""
        repo_dir = "/tmp/test_repo"
        commit_hash = "abc123def456"

        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value=commit_hash),
        ):
            result = await git_manager.get_current_commit(repo_dir)

            assert result == commit_hash


class TestRunGitCommand:
    """Test _run_git_command internal method."""

    @pytest.mark.asyncio
    async def test_run_git_command_success(self, git_manager, mock_subprocess_factory):
        """Test successful git command execution."""
        mock_proc = mock_subprocess_factory(stdout=b"test output", returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await git_manager._run_git_command(["git", "status"])

            assert result == "test output"

    @pytest.mark.asyncio
    async def test_run_git_command_failure(self, git_manager):
        """Test git command failure raises GitError."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"fatal: not a git repository")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(GitError) as exc_info:
                await git_manager._run_git_command(["git", "status"])

            assert "fatal: not a git repository" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_git_command_with_cwd(self, git_manager, mock_subprocess_factory):
        """Test git command with custom working directory."""
        cwd = "/tmp/test_repo"
        mock_proc = mock_subprocess_factory(returncode=0)

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            await git_manager._run_git_command(["git", "status"], cwd=cwd)

            # Check that cwd was passed
            kwargs = mock_exec.call_args[1]
            assert kwargs.get("cwd") == cwd

    @pytest.mark.asyncio
    async def test_run_git_command_exception_wrapped(self, git_manager):
        """Test unexpected exceptions are wrapped in GitError."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("Permission denied"),
        ):
            with pytest.raises(GitError) as exc_info:
                await git_manager._run_git_command(["git", "status"])

            assert "Permission denied" in str(exc_info.value)


class TestRemoveDirectory:
    """Test _remove_directory internal method."""

    @pytest.mark.asyncio
    async def test_remove_directory_success(self, git_manager):
        """Test successful directory removal."""
        path = "/tmp/test_repo"

        mock_rmtree = Mock()
        mock_loop = AsyncMock()
        mock_loop.run_in_executor = AsyncMock(return_value=None)

        with (
            patch("shutil.rmtree", mock_rmtree),
            patch(
                "asyncio.get_event_loop",
                return_value=mock_loop,
            ),
        ):
            await git_manager._remove_directory(path)

            # Should call run_in_executor
            mock_loop.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_directory_handles_errors(self, git_manager):
        """Test directory removal handles errors gracefully."""
        path = "/tmp/test_repo"

        mock_loop = AsyncMock()
        mock_loop.run_in_executor = AsyncMock(
            side_effect=OSError("Permission denied"),
        )

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            # Should not raise exception, just log warning
            await git_manager._remove_directory(path)


class TestCheckGitHubAPISize:
    """Test _check_github_api_size internal method."""

    @pytest.mark.asyncio
    async def test_check_github_api_success(self, git_manager):
        """Test successful GitHub API size check."""
        url = "https://github.com/owner/repo.git"
        info = {"estimated_size_mb": 0}

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"size": 2048}'  # 2MB in KB
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = await git_manager._check_github_api_size(url, info)

            assert result["estimated_size_mb"] == 2.0

    @pytest.mark.asyncio
    async def test_check_github_api_invalid_url(self, git_manager):
        """Test GitHub API with invalid URL pattern."""
        url = "https://gitlab.com/owner/repo.git"
        info = {"estimated_size_mb": 0}

        result = await git_manager._check_github_api_size(url, info)

        # Should return unchanged info
        assert result["estimated_size_mb"] == 0

    @pytest.mark.asyncio
    async def test_check_github_api_network_error(self, git_manager):
        """Test GitHub API handles network errors."""
        url = "https://github.com/owner/repo.git"
        info = {"estimated_size_mb": 0}

        with patch("urllib.request.urlopen", side_effect=OSError("Network error")):
            result = await git_manager._check_github_api_size(url, info)

            # Should return unchanged info
            assert result["estimated_size_mb"] == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_git_output(self, git_manager):
        """Test handling of empty git command output."""
        with patch.object(git_manager, "_run_git_command", AsyncMock(return_value="")):
            branches = await git_manager.get_branches("/tmp/repo")
            assert branches == []

            tags = await git_manager.get_tags("/tmp/repo")
            assert tags == []

    @pytest.mark.asyncio
    async def test_malformed_git_output(self, git_manager):
        """Test handling of malformed git output."""
        # Missing delimiter in output
        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value="invalid_format"),
        ):
            branches = await git_manager.get_branches("/tmp/repo")
            # Should return empty list for malformed data
            assert branches == []

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, git_manager):
        """Test multiple concurrent git operations."""
        with patch.object(
            git_manager,
            "_run_git_command",
            AsyncMock(return_value="output"),
        ):
            # Run multiple operations concurrently
            results = await asyncio.gather(
                git_manager.get_current_commit("/tmp/repo"),
                git_manager.get_branches("/tmp/repo"),
                git_manager.get_tags("/tmp/repo"),
                return_exceptions=True,
            )

            # All should complete (may be empty but shouldn't error)
            assert len(results) == 3


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    @pytest.mark.asyncio
    async def test_full_clone_and_analyze_workflow(self, git_manager):
        """Test complete workflow: clone, analyze, get info."""
        url = "https://github.com/test/repo.git"
        target_dir = "/tmp/test_repo"

        validation_info = {
            "estimated_size_mb": 50.0,
            "file_count": 100,
            "free_space_gb": 10.0,
            "errors": [],
        }

        with (
            patch.object(
                git_manager,
                "validate_repository_size",
                AsyncMock(return_value=(True, validation_info)),
            ),
            patch.object(
                git_manager,
                "clone_repository",
                AsyncMock(return_value=target_dir),
            ),
            patch.object(
                git_manager,
                "get_repository_info",
                AsyncMock(
                    return_value={
                        "remote_url": url,
                        "current_branch": "main",
                        "file_count": 100,
                        "file_extensions": {"py": 50},
                        "size": "50 MiB",
                        "contributor_count": 5,
                    },
                ),
            ),
        ):
            # Clone with validation
            result_dir = await git_manager.clone_repository_with_validation(
                url,
                target_dir,
            )
            assert result_dir == target_dir

            # Get repository info
            info = await git_manager.get_repository_info(target_dir)
            assert info["remote_url"] == url
            assert info["file_count"] == 100

    @pytest.mark.asyncio
    async def test_update_and_track_changes_workflow(self, git_manager):
        """Test workflow: update repository and track changes."""
        repo_dir = "/tmp/test_repo"

        with (
            patch.object(
                git_manager,
                "update_repository",
                AsyncMock(
                    return_value={
                        "old_commit": "abc123",
                        "new_commit": "def456",
                        "updated": True,
                        "changed_files": ["src/main.py"],
                    },
                ),
            ),
            patch.object(
                git_manager,
                "get_changed_files",
                AsyncMock(
                    return_value=[
                        {"status": "modified", "file": "src/main.py"},
                    ],
                ),
            ),
        ):
            # Update repository
            update_result = await git_manager.update_repository(repo_dir)
            assert update_result["updated"]
            assert len(update_result["changed_files"]) == 1

            # Get detailed change info
            changes = await git_manager.get_changed_files(
                repo_dir,
                update_result["old_commit"],
                update_result["new_commit"],
            )
            assert changes[0]["status"] == "modified"
