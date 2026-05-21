#!/usr/bin/env python3
import subprocess
import sys
import re
from datetime import datetime, timedelta


def run_git_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/workspace",
            check=True
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout.strip(), e.stderr.strip()


def get_all_remote_branches():
    # Use git ls-remote --heads to get all remote branches (ignores fetch refspec)
    code, ls_remote_output, stderr = run_git_command(["git", "ls-remote", "--heads", "origin"])
    if code != 0:
        return [], stderr
    branches = []
    for line in ls_remote_output.splitlines():
        if line.strip():
            parts = line.split()
            if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                branches.append(parts[1][len("refs/heads/"):])
    return branches, None

def main():
    # Step 1: Check main branch exists
    code, stdout, stderr = run_git_command(["git", "ls-remote", "--heads", "origin", "main"])
    if code != 0 or not stdout:
        print("Error: origin/main not found")
        sys.exit(1)

    # Step 2: Get current timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    backup_branch = f"backup/main/{timestamp}"
    print(f"Creating backup branch: {backup_branch}")

    # Step 3: Create and push backup branch
    code, stdout, stderr = run_git_command(["git", "push", "origin", f"origin/main:refs/heads/{backup_branch}"])
    if code != 0:
        print(f"Failed to push backup branch: {stderr}")
        sys.exit(1)

    # Get commit hash of origin/main
    code, commit_hash, stderr = run_git_command(["git", "rev-parse", "origin/main"])
    if code != 0:
        print(f"Failed to get commit hash: {stderr}")
        sys.exit(1)
    print(f"Backup created at commit: {commit_hash}")

    # Step 4: Clean up old backups
    print("\nCleaning up old backup branches (older than 30 days)...")
    remote_branches, err = get_all_remote_branches()
    if err:
        print(f"Failed to list remote branches: {err}")
        sys.exit(1)

    # Pattern to match backup branches: backup/main/YYYYMMDD-HHMMSS
    backup_pattern = re.compile(r"^backup/main/(\d{8})-(\d{6})$")
    thirty_days_ago = now - timedelta(days=30)

    deleted_branches = []

    for branch in remote_branches:
        match = backup_pattern.match(branch)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            try:
                branch_datetime = datetime.strptime(f"{date_str}-{time_str}", "%Y%m%d-%H%M%S")
                if branch_datetime < thirty_days_ago:
                    # Delete the branch
                    print(f"Deleting old branch: {branch}")
                    code, _, stderr = run_git_command(["git", "push", "origin", "--delete", branch])
                    if code != 0:
                        print(f"Failed to delete {branch}: {stderr}")
                    else:
                        deleted_branches.append(branch)
            except ValueError:
                # Not a valid date, skip
                pass

    # Now get all remaining backup branches
    remote_branches, err = get_all_remote_branches()
    if err:
        print(f"Failed to list remote branches: {err}")
        sys.exit(1)

    remaining_backups = []
    for branch in remote_branches:
        match = backup_pattern.match(branch)
        if match:
            remaining_backups.append(branch)

    # Step 5: Report results
    print("\n--- Backup Result ---")
    print(f"Created backup branch: {backup_branch}")
    print(f"Backup commit hash: {commit_hash}")

    print("\n--- Deleted Branches ---")
    if deleted_branches:
        for b in deleted_branches:
            print(b)
    else:
        print("None")

    print("\n--- Remaining Backup Branches ---")
    if remaining_backups:
        for b in remaining_backups:
            print(b)
    else:
        print("None")


if __name__ == "__main__":
    main()
