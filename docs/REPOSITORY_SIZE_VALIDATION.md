# Repository Size Validation

## Overview

The Crawl4AI MCP server includes comprehensive repository size validation to prevent resource exhaustion when parsing large repositories. This feature helps protect your system from accidentally cloning extremely large repositories that could consume excessive disk space, memory, or processing time.

## Features

- **Pre-clone validation**: Checks repository size before cloning
- **Disk space validation**: Ensures sufficient free space is available
- **File count limits**: Prevents processing repositories with too many files
- **Configurable limits**: Adjust limits based on your deployment environment
- **Override capability**: Force processing when needed with proper authorization
- **Multiple detection methods**: Uses shallow clones and GitHub API for accurate size estimation

## Configuration

Configure repository limits using environment variables:

```bash
# Maximum repository size in MB (default: 500)
REPO_MAX_SIZE_MB=500

# Maximum number of files (default: 10000)
REPO_MAX_FILE_COUNT=10000

# Minimum free disk space required in GB (default: 1.0)
REPO_MIN_FREE_SPACE_GB=1.0

# Allow overriding size limits (default: false)
REPO_ALLOW_SIZE_OVERRIDE=false
```

## Usage Examples

### Basic Repository Parsing

```python
# The validation happens automatically
from src.knowledge_graph.parse_repo_into_neo4j import DirectNeo4jExtractor

extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)
await extractor.initialize()

# Will validate size before cloning
await extractor.analyze_repository("https://github.com/user/repo.git")
```

### Force Processing Large Repository

```python
# Override validation for specific repository
await extractor.analyze_repository(
    "https://github.com/large/repository.git",
    force=True  # Override size validation
)
```

### Manual Validation

```python
from src.knowledge_graph.git_manager import GitRepositoryManager

manager = GitRepositoryManager()

# Check repository size before processing
is_valid, info = await manager.validate_repository_size(
    url="https://github.com/user/repo.git",
    max_size_mb=500,
    max_file_count=10000,
    min_free_space_gb=1.0
)

if is_valid:
    print(f"Repository is safe to clone: {info['estimated_size_mb']:.2f}MB")
else:
    print(f"Repository exceeds limits: {', '.join(info['errors'])}")
```

## Validation Process

1. **Disk Space Check**: Verifies available disk space meets minimum requirements
2. **Shallow Clone**: Performs minimal clone to estimate repository size
3. **GitHub API**: Falls back to GitHub API for size information if available
4. **Limit Validation**: Checks against configured size and file count limits
5. **Safety Margin**: Ensures 2x the repository size is available as free space

## Error Messages

When validation fails, you'll receive clear error messages:

- `"Repository too large: 750.00MB exceeds limit of 500MB"`
- `"Too many files: 15000 exceeds limit of 10000"`
- `"Insufficient disk space: 0.50GB available, 1.00GB required"`
- `"Insufficient space for repository: 2.00GB needed, 1.50GB available"`

## Best Practices

### Development Environment

```bash
# Relaxed limits for development
REPO_MAX_SIZE_MB=1000
REPO_MAX_FILE_COUNT=20000
REPO_MIN_FREE_SPACE_GB=0.5
```

### Production Environment

```bash
# Strict limits for production
REPO_MAX_SIZE_MB=250
REPO_MAX_FILE_COUNT=5000
REPO_MIN_FREE_SPACE_GB=2.0
REPO_ALLOW_SIZE_OVERRIDE=false
```

### CI/CD Environment

```bash
# Moderate limits for CI/CD
REPO_MAX_SIZE_MB=500
REPO_MAX_FILE_COUNT=10000
REPO_MIN_FREE_SPACE_GB=1.0
```

## Performance Considerations

- **Validation overhead**: Adds 1-3 seconds to repository processing
- **Shallow clone**: Uses `--depth 1` and `--bare` for minimal transfer
- **Caching**: GitHub API results can be cached for repeated validations
- **Parallel processing**: Multiple repositories can be validated concurrently

## Troubleshooting

### Repository shows as 0MB

Some repositories may show 0MB during validation if:

- The repository uses Git LFS for large files
- The shallow clone doesn't include full history
- GitHub API rate limits are exceeded

Solution: Use the GitHub API method or increase clone depth.

### Validation passes but clone fails

This can happen if:

- Network issues occur during actual cloning
- Repository permissions change
- Disk space is consumed by other processes

Solution: Monitor disk space and retry with updated limits.

### Override not working

Ensure:

- `REPO_ALLOW_SIZE_OVERRIDE=true` is set
- The `force=True` parameter is passed to the method
- User has appropriate permissions

## Security Considerations

- **Resource exhaustion**: Prevents DoS through large repository submissions
- **Disk space protection**: Maintains system stability
- **Override logging**: All overrides are logged for audit purposes
- **Rate limiting**: Consider implementing rate limits for validation requests

## Future Enhancements

- [ ] Support for Git LFS size calculation
- [ ] Redis caching for validation results
- [ ] Progressive size estimation during clone
- [ ] Support for other Git platforms (GitLab, Bitbucket)
- [ ] Per-user or per-organization limits
- [ ] Webhook notifications for limit violations
