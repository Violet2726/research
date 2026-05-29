#!/usr/bin/env python3
import subprocess
import re
from datetime import datetime, timedelta
import sys

def run_command(cmd, cwd='/workspace'):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip(), None
    except subprocess.CalledProcessError as e:
        return None, f"Command failed: {cmd}\nError: {e.stderr.strip()}"

def main():
    print("=== 主分支备份与清理任务 ===")
    
    # Step 1: Prepare and check
    print("\n--- 准备与检查 ---")
    main_branch = 'origin/main'
    commit_hash, error = run_command(f"git rev-parse {main_branch}")
    if error:
        print(f"错误: {error}")
        sys.exit(1)
    print(f"✓ 主分支: {main_branch}")
    print(f"✓ 最新 commit: {commit_hash}")
    
    # Step 2: Create backup branch
    print("\n--- 创建备份分支 ---")
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    backup_branch_name = f"backup/main/{timestamp}"
    print(f"备份分支: {backup_branch_name}")
    
    # Push backup branch
    _, error = run_command(f"git push origin {commit_hash}:refs/heads/{backup_branch_name}")
    if error:
        print(f"错误: {error}")
        sys.exit(1)
    print(f"✓ 备份分支已创建并推送")
    
    # Step 3: Clean up old backups
    print("\n--- 清理旧备份 ---")
    
    # Use ls-remote to get all remote branches
    ls_remote_output, error = run_command("git ls-remote --heads origin")
    if error:
        print(f"错误: {error}")
        sys.exit(1)
    
    backup_pattern = re.compile(r'refs/heads/backup/main/(\d{8})-(\d{6})$')
    backup_branches = []
    for line in ls_remote_output.split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:
            ref = parts[1]
            match = backup_pattern.match(ref)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                try:
                    branch_date = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
                    branch_name = ref.replace('refs/heads/', '')
                    backup_branches.append((branch_name, branch_date))
                except ValueError:
                    continue
    
    print(f"✓ 找到 {len(backup_branches)} 个备份分支")
    
    cutoff = now - timedelta(days=30)
    print(f"✓ 清理截止日期: {cutoff.strftime('%Y%m%d-%H%M%S')}")
    
    branches_to_delete = []
    branches_to_keep = []
    
    for branch_name, branch_date in backup_branches:
        if branch_date < cutoff:
            branches_to_delete.append(branch_name)
        else:
            branches_to_keep.append(branch_name)
    
    deleted_branches = []
    if branches_to_delete:
        print(f"\n准备删除 {len(branches_to_delete)} 个旧备份分支:")
        for branch in branches_to_delete:
            print(f"  - {branch}")
            _, error = run_command(f"git push origin --delete {branch}")
            if error:
                print(f"    警告: 删除失败: {error}")
            else:
                deleted_branches.append(branch)
    else:
        print("✓ 没有需要删除的旧备份分支")
    
    # Refresh and verify
    print("\n--- 验证结果 ---")
    ls_remote_output, _ = run_command("git ls-remote --heads origin")
    current_backups = []
    for line in ls_remote_output.split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:
            ref = parts[1]
            if backup_pattern.match(ref):
                current_backups.append(ref.replace('refs/heads/', ''))
    current_backups.sort()
    
    # Final report
    print("\n" + "="*70)
    print("任务完成报告")
    print("="*70)
    print(f"1. 本次创建的备份分支名称: {backup_branch_name}")
    print(f"2. 备份对应的 commit hash: {commit_hash}")
    print(f"3. 本次删除的旧备份分支列表:")
    if deleted_branches:
        for b in deleted_branches:
            print(f"   - {b}")
    else:
        print("   - 无")
    print(f"4. 当前仍保留的备份分支列表:")
    if current_backups:
        for b in current_backups:
            print(f"   - {b}")
    else:
        print("   - 无")
    print("="*70)

if __name__ == "__main__":
    main()
