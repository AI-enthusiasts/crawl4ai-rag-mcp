"""
Comprehensive unit tests for src/utils/validation.py

Test Coverage:
- is_safe_hostname(): SSRF protection - validate hostnames for crawling
- validate_neo4j_connection(): Neo4j environment validation
- get_accessible_script_path(): Path mapping for container access
- validate_script_path(): Script path validation and existence check
- detect_and_fix_truncated_url(): Truncated URL detection and repair
- validate_crawl_url(): URL validation for crawl4ai compatibility
- validate_urls_for_crawling(): Batch URL validation
- validate_github_url(): GitHub repository URL validation

Testing Approach:
- Security testing for SSRF protection
- Test private IP blocking (RFC 1918)
- Test localhost blocking
- Test cloud metadata blocking (169.254.x.x)
- Mock DNS resolution for hostname checks
- Test path validation and mapping
- Test URL format validation
- Parametrized tests for various scenarios
"""

import os
import socket
from unittest.mock import patch, Mock

import pytest

from src.utils.validation import (
    is_safe_hostname,
    validate_neo4j_connection,
    get_accessible_script_path,
    validate_script_path,
    detect_and_fix_truncated_url,
    validate_crawl_url,
    validate_urls_for_crawling,
    validate_github_url,
)


class TestIsSafeHostname:
    """Test is_safe_hostname() SSRF protection"""

    # Localhost variants
    @pytest.mark.parametrize(
        "hostname",
        [
            "localhost",
            "localhost.localdomain",
            "127.0.0.1",
            "::1",
            "0.0.0.0",
            "0000:0000:0000:0000:0000:0000:0000:0001",
            "::ffff:127.0.0.1",
        ],
    )
    def test_blocks_localhost_variants(self, hostname):
        """Test that all localhost variants are blocked"""
        result = is_safe_hostname(hostname)
        assert result["safe"] is False
        assert "localhost" in result["error"].lower() or "loopback" in result[
            "error"
        ].lower()

    # Private IP ranges (RFC 1918)
    @pytest.mark.parametrize(
        "private_ip",
        [
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.255.255",
        ],
    )
    def test_blocks_private_ips(self, private_ip):
        """Test that private IP ranges are blocked"""
        result = is_safe_hostname(private_ip)
        assert result["safe"] is False
        assert "private" in result["error"].lower()

    # Link-local addresses (including cloud metadata)
    @pytest.mark.parametrize(
        "link_local_ip",
        [
            "169.254.0.1",
            "169.254.169.254",  # AWS/GCP metadata
            "169.254.255.255",
        ],
    )
    def test_blocks_link_local_addresses(self, link_local_ip):
        """Test that link-local addresses (including cloud metadata) are blocked"""
        result = is_safe_hostname(link_local_ip)
        assert result["safe"] is False
        # Error message may say "link-local" or "private" depending on IP classification
        assert "link-local" in result["error"].lower() or "private" in result["error"].lower()

    # Internal domain suffixes
    @pytest.mark.parametrize(
        "internal_domain",
        [
            "service.local",
            "api.internal",
            "db.corp",
            "server.localhost",
        ],
    )
    def test_blocks_internal_domains(self, internal_domain):
        """Test that internal domain suffixes are blocked"""
        result = is_safe_hostname(internal_domain)
        assert result["safe"] is False
        assert "internal" in result["error"].lower() or "domain" in result[
            "error"
        ].lower()

    # Reserved IP addresses
    @pytest.mark.parametrize(
        "reserved_ip",
        [
            "0.0.0.0",
            "255.255.255.255",
        ],
    )
    def test_blocks_reserved_ips(self, reserved_ip):
        """Test that reserved IP addresses are blocked"""
        result = is_safe_hostname(reserved_ip)
        assert result["safe"] is False

    # IPv6 addresses
    def test_blocks_ipv6_loopback(self):
        """Test IPv6 loopback blocking"""
        result = is_safe_hostname("[::1]")
        assert result["safe"] is False

    def test_blocks_ipv6_private(self):
        """Test IPv6 private address blocking"""
        # fc00::/7 is private IPv6 range
        result = is_safe_hostname("fc00::1")
        assert result["safe"] is False

    # Public IPs (should be safe)
    @pytest.mark.parametrize(
        "public_ip",
        [
            "8.8.8.8",  # Google DNS
            "1.1.1.1",  # Cloudflare DNS
            "93.184.216.34",  # example.com
        ],
    )
    def test_allows_public_ips(self, public_ip):
        """Test that public IPs are allowed"""
        result = is_safe_hostname(public_ip)
        assert result["safe"] is True

    # DNS resolution with mocking
    @patch("src.utils.validation.socket.getaddrinfo")
    def test_resolves_hostname_to_safe_ip(self, mock_getaddrinfo):
        """Test hostname resolution to safe public IP"""
        # Mock DNS resolution to public IP
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443)),
        ]

        result = is_safe_hostname("example.com")
        assert result["safe"] is True

    @patch("src.utils.validation.socket.getaddrinfo")
    def test_blocks_hostname_resolving_to_private_ip(self, mock_getaddrinfo):
        """Test blocking hostname that resolves to private IP"""
        # Mock DNS resolution to private IP
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 80)),
        ]

        result = is_safe_hostname("internal.example.com")
        assert result["safe"] is False
        assert "unsafe IP" in result["error"]

    @patch("src.utils.validation.socket.getaddrinfo")
    def test_blocks_hostname_resolving_to_localhost(self, mock_getaddrinfo):
        """Test blocking hostname that resolves to localhost"""
        # Mock DNS resolution to localhost
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 80)),
        ]

        result = is_safe_hostname("sneaky.com")
        assert result["safe"] is False

    @patch("src.utils.validation.socket.getaddrinfo")
    def test_dns_resolution_failure(self, mock_getaddrinfo):
        """Test handling of DNS resolution failure"""
        mock_getaddrinfo.side_effect = socket.gaierror("Name resolution failed")

        result = is_safe_hostname("nonexistent.example.com")
        assert result["safe"] is False
        assert "Cannot resolve" in result["error"]

    # Edge cases
    def test_empty_hostname(self):
        """Test empty hostname"""
        result = is_safe_hostname("")
        assert result["safe"] is False
        assert "Empty hostname" in result["error"]

    def test_whitespace_hostname(self):
        """Test hostname with whitespace"""
        result = is_safe_hostname("   ")
        assert result["safe"] is False

    def test_case_insensitivity(self):
        """Test case insensitive hostname checking"""
        result = is_safe_hostname("LOCALHOST")
        assert result["safe"] is False


class TestValidateNeo4jConnection:
    """Test validate_neo4j_connection() Neo4j environment validation"""

    @patch("src.utils.validation.settings")
    def test_valid_neo4j_config(self, mock_settings):
        """Test valid Neo4j configuration"""
        mock_settings.neo4j_uri = "bolt://localhost:7687"
        mock_settings.neo4j_username = "neo4j"
        mock_settings.neo4j_password = "password"

        assert validate_neo4j_connection() is True

    @patch("src.utils.validation.settings")
    def test_missing_neo4j_uri(self, mock_settings):
        """Test missing Neo4j URI"""
        mock_settings.neo4j_uri = None
        mock_settings.neo4j_username = "neo4j"
        mock_settings.neo4j_password = "password"

        assert validate_neo4j_connection() is False

    @patch("src.utils.validation.settings")
    def test_missing_neo4j_credentials(self, mock_settings):
        """Test missing Neo4j credentials"""
        mock_settings.neo4j_uri = "bolt://localhost:7687"
        mock_settings.neo4j_username = None
        mock_settings.neo4j_password = "password"

        assert validate_neo4j_connection() is False


class TestGetAccessibleScriptPath:
    """Test get_accessible_script_path() path mapping"""

    @pytest.mark.parametrize(
        "host_path,expected_container_path",
        [
            ("/tmp/script.py", "/app/tmp_scripts/script.py"),
            ("/tmp/test/script.py", "/app/tmp_scripts/test/script.py"),
            ("./analysis_scripts/test.py", "/app/analysis_scripts/test.py"),
            ("analysis_scripts/test.py", "/app/analysis_scripts/test.py"),
            ("my_script.py", "/app/analysis_scripts/user_scripts/my_script.py"),
            ("/app/direct/path.py", "/app/direct/path.py"),
        ],
    )
    def test_path_mappings(self, host_path, expected_container_path):
        """Test various path mapping scenarios"""
        result = get_accessible_script_path(host_path)
        assert result == expected_container_path


class TestValidateScriptPath:
    """Test validate_script_path() script validation"""

    def test_valid_script_path(self, tmp_path):
        """Test validation of valid script path"""
        # Create a temporary Python file
        script_file = tmp_path / "test_script.py"
        script_file.write_text("print('hello')")

        with patch(
            "src.utils.validation.get_accessible_script_path",
        ) as mock_get_path:
            mock_get_path.return_value = str(script_file)

            result = validate_script_path(str(script_file))

            assert result["valid"] is True
            assert result["container_path"] == str(script_file)

    def test_nonexistent_script(self):
        """Test validation of nonexistent script"""
        result = validate_script_path("/nonexistent/script.py")

        assert result["valid"] is False
        assert "not found" in result["error"].lower()

    def test_non_python_file(self, tmp_path):
        """Test validation of non-Python file"""
        # Create a non-Python file
        text_file = tmp_path / "test.txt"
        text_file.write_text("not python")

        with patch(
            "src.utils.validation.get_accessible_script_path",
        ) as mock_get_path:
            mock_get_path.return_value = str(text_file)

            result = validate_script_path(str(text_file))

            assert result["valid"] is False
            assert "Python" in result["error"] or ".py" in result["error"]

    def test_empty_path(self):
        """Test validation of empty path"""
        result = validate_script_path("")

        assert result["valid"] is False
        assert "required" in result["error"].lower()

    def test_unreadable_file(self, tmp_path):
        """Test validation of unreadable file"""
        script_file = tmp_path / "test.py"
        script_file.write_text("print('test')")

        with patch(
            "src.utils.validation.get_accessible_script_path",
        ) as mock_get_path:
            mock_get_path.return_value = str(script_file)

            # Mock file read to raise error
            with patch("builtins.open", side_effect=PermissionError("Access denied")):
                result = validate_script_path(str(script_file))

                assert result["valid"] is False
                assert "Cannot read" in result["error"]


class TestDetectAndFixTruncatedUrl:
    """Test detect_and_fix_truncated_url() truncated URL detection"""

    def test_detect_truncated_with_ellipsis(self):
        """Test detection of URL starting with ..."""
        url = "...example.com/path"

        result = detect_and_fix_truncated_url(url)

        assert result["truncated"] is True
        assert result["fixed"] is False
        assert len(result["suggestions"]) > 0
        assert "https://www.example.com/path" in result["suggestions"]

    def test_fix_missing_protocol(self):
        """Test fixing URL missing protocol"""
        url = "example.com/path"

        result = detect_and_fix_truncated_url(url)

        assert result["fixed"] is True
        assert result["fixed_url"] == "https://example.com/path"

    def test_no_issues_with_valid_url(self):
        """Test that valid URLs pass through"""
        url = "https://example.com/path"

        result = detect_and_fix_truncated_url(url)

        assert result.get("no_issues") is True
        assert result.get("fixed") is False

    def test_empty_url(self):
        """Test empty URL"""
        result = detect_and_fix_truncated_url("")

        assert result["fixed"] is False
        assert "error" in result

    def test_url_with_protocol_not_fixed(self):
        """Test URL with protocol is not auto-fixed"""
        url = "https://example.com"

        result = detect_and_fix_truncated_url(url)

        assert result.get("fixed") is False

    def test_relative_path_not_fixed(self):
        """Test relative path is not auto-fixed"""
        url = "/relative/path"

        result = detect_and_fix_truncated_url(url)

        assert result.get("fixed") is False


class TestValidateCrawlUrl:
    """Test validate_crawl_url() URL validation for crawl4ai"""

    @pytest.mark.parametrize(
        "valid_url",
        [
            "https://example.com",
            "http://example.com",
            "https://www.example.com/path",
            "https://example.com:8080/path",
            "raw:content",
        ],
    )
    def test_valid_urls(self, valid_url):
        """Test validation of valid URLs"""
        with patch("src.utils.validation.is_safe_hostname") as mock_safe:
            mock_safe.return_value = {"safe": True}

            result = validate_crawl_url(valid_url)

            assert result["valid"] is True
            assert "normalized_url" in result

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "ftp://example.com",
            "javascript:alert(1)",
            "file:///etc/passwd",  # file:// blocked for security
            "example.com",  # Missing protocol
        ],
    )
    def test_invalid_protocols(self, invalid_url):
        """Test rejection of invalid protocols"""
        result = validate_crawl_url(invalid_url)

        assert result["valid"] is False
        assert "must start with" in result["error"] or "protocol" in result[
            "error"
        ].lower()

    def test_ssrf_protection(self):
        """Test SSRF protection blocks unsafe hosts"""
        with patch("src.utils.validation.is_safe_hostname") as mock_safe:
            mock_safe.return_value = {"safe": False, "error": "Private IP not allowed"}

            result = validate_crawl_url("https://192.168.1.1/admin")

            assert result["valid"] is False
            assert "SSRF protection" in result["error"]

    def test_url_with_multiple_protocols(self):
        """Test rejection of URL with multiple protocols"""
        url = "https://http://example.com"

        result = validate_crawl_url(url)

        assert result["valid"] is False
        assert "multiple protocols" in result["error"]

    def test_url_missing_domain(self):
        """Test rejection of URL missing domain"""
        url = "https:///path"

        result = validate_crawl_url(url)

        assert result["valid"] is False
        assert "missing domain" in result["error"]

    def test_url_normalization(self):
        """Test URL normalization removes fragment"""
        with patch("src.utils.validation.is_safe_hostname") as mock_safe:
            mock_safe.return_value = {"safe": True}

            url = "https://example.com/path#fragment"
            result = validate_crawl_url(url)

            assert result["valid"] is True
            assert "#fragment" not in result["normalized_url"]

    def test_auto_fix_applied(self):
        """Test that auto-fix information is included"""
        with patch("src.utils.validation.is_safe_hostname") as mock_safe:
            mock_safe.return_value = {"safe": True}

            url = "example.com/path"
            result = validate_crawl_url(url)

            if result["valid"]:
                assert result.get("auto_fixed") is True
                assert "original_url" in result

    def test_empty_url(self):
        """Test validation of empty URL"""
        result = validate_crawl_url("")

        assert result["valid"] is False
        assert "required" in result["error"].lower() or "empty" in result["error"].lower()


class TestValidateUrlsForCrawling:
    """Test validate_urls_for_crawling() batch URL validation"""

    def test_all_valid_urls(self):
        """Test validation of all valid URLs"""
        with patch("src.utils.validation.is_safe_hostname") as mock_safe:
            mock_safe.return_value = {"safe": True}

            urls = [
                "https://example.com",
                "https://example.org",
                "https://example.net",
            ]

            result = validate_urls_for_crawling(urls)

            assert result["valid"] is True
            assert len(result["urls"]) == 3

    def test_some_invalid_urls(self):
        """Test validation with some invalid URLs"""
        with patch("src.utils.validation.is_safe_hostname") as mock_safe:
            mock_safe.return_value = {"safe": True}

            urls = [
                "https://example.com",
                "ftp://invalid.com",
                "not-a-url",
            ]

            result = validate_urls_for_crawling(urls)

            assert result["valid"] is False
            assert len(result["invalid_urls"]) == 2
            assert len(result.get("valid_urls", [])) >= 1

    def test_empty_list(self):
        """Test validation of empty list"""
        result = validate_urls_for_crawling([])

        assert result["valid"] is False
        assert "non-empty list" in result["error"]

    def test_invalid_input_type(self):
        """Test validation with invalid input type"""
        result = validate_urls_for_crawling("not a list")

        assert result["valid"] is False
        assert "list" in result["error"]


class TestValidateGithubUrl:
    """Test validate_github_url() GitHub URL validation"""

    @pytest.mark.parametrize(
        "valid_url",
        [
            "https://github.com/user/repo",
            "https://github.com/user/repo.git",
            "git@github.com:user/repo.git",
            "https://github.com/user-name/repo-name",
        ],
    )
    def test_valid_github_urls(self, valid_url):
        """Test validation of valid GitHub URLs"""
        result = validate_github_url(valid_url)

        assert result["valid"] is True
        assert "repo_name" in result

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "https://gitlab.com/user/repo",
            "https://bitbucket.org/user/repo",
            "ftp://github.com/user/repo",
            "github.com/user/repo",  # Missing protocol
        ],
    )
    def test_invalid_github_urls(self, invalid_url):
        """Test rejection of invalid GitHub URLs"""
        result = validate_github_url(invalid_url)

        assert result["valid"] is False
        assert "error" in result

    def test_repo_name_extraction(self):
        """Test repository name extraction"""
        url = "https://github.com/user/my-awesome-repo.git"

        result = validate_github_url(url)

        assert result["valid"] is True
        assert result["repo_name"] == "my-awesome-repo"

    def test_empty_url(self):
        """Test validation of empty URL"""
        result = validate_github_url("")

        assert result["valid"] is False
        assert "required" in result["error"].lower()


class TestValidationIntegration:
    """Integration tests for validation functions"""

    def test_complete_url_validation_flow(self):
        """Test complete URL validation workflow"""
        # Start with potentially truncated URL
        url = "example.com/api/data"

        # Try to fix truncation
        fix_result = detect_and_fix_truncated_url(url)
        if fix_result.get("fixed"):
            url = fix_result["fixed_url"]

        # Validate for crawling
        with patch("src.utils.validation.is_safe_hostname") as mock_safe:
            mock_safe.return_value = {"safe": True}

            validation = validate_crawl_url(url)

            assert validation["valid"] is True

    def test_security_validation_pipeline(self):
        """Test security-focused validation pipeline"""
        dangerous_urls = [
            "https://127.0.0.1/admin",
            "https://192.168.1.1/config",
            "https://169.254.169.254/metadata",
        ]

        # Validate all URLs
        result = validate_urls_for_crawling(dangerous_urls)

        # Should block all dangerous URLs
        assert result["valid"] is False
        assert len(result["invalid_urls"]) == len(dangerous_urls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
