"""
Security check script - Verify no secrets are being committed to Git.

Run this before committing to ensure no sensitive data is included.
"""
import subprocess
import sys
import re


# Patterns that might indicate secrets
SECRET_PATTERNS = [
    (r'strava_client_secret:\s*str\s*=\s*["\']([a-f0-9]{40})["\']', 'Strava Client Secret (hardcoded in settings)'),
    (r'strava_refresh_token:\s*str\s*=\s*["\']([a-f0-9]{40})["\']', 'Strava Refresh Token (hardcoded in settings)'),
    (r'strava_client_id:\s*str\s*=\s*["\'](\d+)["\']', 'Strava Client ID (hardcoded in settings)'),
    (r'STRAVA_CLIENT_SECRET.*=.*["\']([a-f0-9]{40})["\']', 'Strava Client Secret'),
    (r'STRAVA_REFRESH_TOKEN.*=.*["\']([a-f0-9]{40})["\']', 'Strava Refresh Token'),
    (r'AccountKey=([A-Za-z0-9+/=]{88})', 'Azure Storage Account Key'),
    (r'DefaultEndpointsProtocol=https.*EndpointSuffix', 'Azure Storage Connection String'),
]

# Files that should NOT be committed
FORBIDDEN_FILES = [
    'local.settings.json',
    '.env',
    '.env.local',
]


def check_git_status():
    """Check files staged for commit."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        print("⚠️  Not a git repository or git not installed")
        return []


def check_file_for_secrets(filepath):
    """Check a single file for potential secrets."""
    issues = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        for pattern, secret_type in SECRET_PATTERNS:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append(f"  ⚠️  Line {line_num}: Potential {secret_type} detected")

    except Exception as e:
        print(f"  ℹ️  Could not read {filepath}: {e}")

    return issues


def main():
    """Main security check."""
    print("=" * 80)
    print("🔐 SECURITY CHECK - Scanning for secrets before commit")
    print("=" * 80)
    print()

    staged_files = check_git_status()

    if not staged_files or staged_files == ['']:
        print("ℹ️  No staged files found. Run 'git add' first.")
        return 0

    print(f"Checking {len(staged_files)} staged file(s)...\n")

    errors_found = False

    # Check for forbidden files
    for forbidden_file in FORBIDDEN_FILES:
        for staged_file in staged_files:
            if forbidden_file in staged_file:
                print(f"❌ FORBIDDEN FILE: {staged_file}")
                print(f"   This file contains secrets and should NOT be committed!")
                print(f"   Fix: git reset HEAD {staged_file}\n")
                errors_found = True

    # Check file contents for secrets
    for filepath in staged_files:
        if filepath.endswith(('.py', '.json', '.md', '.txt', '.yml', '.yaml')):
            issues = check_file_for_secrets(filepath)
            if issues:
                print(f"⚠️  {filepath}:")
                for issue in issues:
                    print(issue)
                print()
                errors_found = True

    print("=" * 80)
    if errors_found:
        print("❌ SECURITY CHECK FAILED!")
        print("=" * 80)
        print()
        print("Potential secrets detected in staged files.")
        print("Please review the files above and remove sensitive data.")
        print()
        print("Common fixes:")
        print("  1. Move secrets to environment variables")
        print("  2. Add files to .gitignore")
        print("  3. Use local.settings.json.example instead of local.settings.json")
        print()
        print("See SECURITY_GUIDE.md for more information.")
        print()
        return 1
    else:
        print("✅ SECURITY CHECK PASSED!")
        print("=" * 80)
        print()
        print("No obvious secrets detected in staged files.")
        print("You can proceed with the commit.")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
