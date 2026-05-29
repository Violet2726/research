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
    # Step 1: Prepare and check
    print("=== 准备与检查 ===")
    
    # Check git status and remote
    remote_output, error = run_command("git remote -v")
    if error:
        print(f"错误: {error}")
        sys.exit(1)
    
    # Check origin/main exists
    branch_output, error = run_command("git branch -r")
    if error:
        print(f"错误: {error}")
        sys.exit(1)
    
    remote_branches = branch_output.split('\n')
    main_branch = None
    if '  origin/main' in remote_branches:
        main_branch = 'origin/main'
        print(f"✓ 找到远程主分支: {main_branch}")
    elif '  origin/master' in remote_branches:
        main_branch = 'origin/master'
        print(f"✓ 找到远程主分支: {main_branch}")
    else:
        print("错误: 未找到 origin/main 或 origin/master 分支")
        sys.exit(1)
    
    # Get commit hash of main branch
    commit_hash, error = run_command(f"git rev-parse {main_branch}")
    if error:
        print(f"错误: 无法获取 {main_branch} 的 commit hash: {error}")
        sys.exit(1)
    print(f"✓ 主分支最新 commit: {commit_hash}")
    
    # Step 2: Create backup branch
    print("\n=== 创建备份分支 ===")
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    backup_branch_name = f"backup/main/{timestamp}"
    print(f"备份分支名称: {backup_branch_name}")
    
    # Push backup branch directly
    _, error = run_command(f"git push origin {commit_hash}:refs/heads/{backup_branch_name}")
    if error:
        print(f"错误: 无法推送备份分支: {error}")
        sys.exit(1)
    print("✓ 备份分支已推送")
    
    # Verify backup branch exists
    branch_output, error = run_command("git fetch origin --prune && git branch -r")
    if error:
        print(f"警告: 无法验证备份分支: {error}")
    else:
        if f'  origin/{backup_branch_name}' in branch_output:
            print(f"✓ 验证成功: 远程备份分支已存在")
        else:
            print(f"警告: 无法确认远程备份分支是否存在")
    
    # Step 3: Clean up old backup branches
    print("\n=== 清理旧备份分支 ===")
    branch_output, error = run_command("git branch -r")
    if error:
        print(f"错误: 无法获取远程分支列表: {error}")
        sys.exit(1)
    
    # Pattern to match backup branches: origin/backup/main/YYYYMMDD-HHMMSS
    backup_pattern = re.compile(r'^  origin/backup/main/(\d{8})-(\d{6})$')
    backup_branches = []
    for branch in branch_output.split('\n'):
        branch = branch.strip()
        match = backup_pattern.match(branch)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            try:
                branch_date = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
                backup_branches.append((branch, branch_date, date_str + '-' + time_str))
            except ValueError:
                continue
    
    print(f"找到 {len(backup_branches)} 个备份分支")
    
    # Calculate cutoff date (30 days ago)
    cutoff = now - timedelta(days=30)
    print(f"清理截止日期: {cutoff.strftime('%Y%m%d-%H%M%S')}")
    
    branches_to_delete = []
    branches_to_keep = []
    
    for branch, branch_date, timestamp_str in backup_branches:
        if branch_date < cutoff:
            branches_to_delete.append((branch, timestamp_str))
        else:
            branches_to_keep.append((branch, timestamp_str))
    
    # Delete old branches
    deleted_branches = []
    for branch, timestamp_str in branches_to_delete:
        branch_name = branch.replace('  origin/', '')
        print(f"删除: {branch_name}")
        _, error = run_command(f"git push origin --delete {branch_name}")
        if error:
            print(f"警告: 删除失败 {branch_name}: {error}")
        else:
            deleted_branches.append(branch_name)
    
    # Fetch prune to update remote branches
    run_command("git fetch origin --prune")
    
    # Step 4: Report results
    print("\n" + "="*50)
    print("任务完成报告")
    print("="*50)
    print(f"1. 本次创建的备份分支名称: {backup_branch_name}")
    print(f"2. 备份对应的 commit hash: {commit_hash}")
    print(f"3. 本次删除的旧备份分支列表:")
    if deleted_branches:
        for b in deleted_branches:
            print(f"   - {b}")
    else:
        print("   - 无")
    print(f"4. 当前仍保留的备份分支列表:")
    # Refresh branch list
    branch_output, _ = run_command("git branch -r")
    current_backups = []
    for branch in branch_output.split('\n'):
        match = backup_pattern.match(branch)
        if match:
            current_backups.append(branch.replace('  origin/', ''))
    if current_backups:
        current_backups.sort()
        for b in current_backups:
            print(f"   - {b}")
    else:
        print("   - 无")
    print("="*50)

if __name__ == "__main__":
    main()
