#!/usr/bin/env python3
import subprocess
import re
import datetime
import sys
from typing import List, Tuple


def run_command(cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
    """执行命令并返回退出码、标准输出和标准错误"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout.strip(), e.stderr.strip()


def main():
    print("=" * 80)
    print("Git 主分支备份与旧备份清理任务")
    print("=" * 80)
    
    # 确定主分支
    print("\n[步骤 1] 检查远程主分支...")
    code, stdout, stderr = run_command(["git", "branch", "-r"])
    if code != 0:
        print(f"错误: 无法获取远程分支列表: {stderr}")
        sys.exit(1)
    
    remote_branches = stdout.split()
    main_branch = None
    if "origin/main" in remote_branches:
        main_branch = "origin/main"
        print(f"找到主分支: {main_branch}")
    elif "origin/master" in remote_branches:
        main_branch = "origin/master"
        print(f"找到主分支: {main_branch}")
    else:
        print("错误: 未找到 origin/main 或 origin/master")
        sys.exit(1)
    
    # 获取主分支的最新 commit hash
    code, stdout, stderr = run_command(["git", "rev-parse", main_branch])
    if code != 0:
        print(f"错误: 无法获取主分支 commit hash: {stderr}")
        sys.exit(1)
    main_commit_hash = stdout
    print(f"主分支最新 commit: {main_commit_hash}")
    
    # 生成备份分支名
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_branch = f"backup/main/{timestamp}"
    print(f"\n[步骤 2] 创建备份分支: {backup_branch}")
    
    # 创建本地备份分支
    code, stdout, stderr = run_command(["git", "branch", backup_branch, main_commit_hash])
    if code != 0:
        print(f"错误: 无法创建本地备份分支: {stderr}")
        sys.exit(1)
    
    # 推送到 origin
    code, stdout, stderr = run_command(["git", "push", "origin", backup_branch])
    if code != 0:
        print(f"错误: 无法推送备份分支到远程: {stderr}")
        # 清理本地分支以免影响后续
        run_command(["git", "branch", "-D", backup_branch], check=False)
        sys.exit(1)
    
    print(f"成功推送备份分支到远程: {backup_branch}")
    
    # 验证备份是否成功
    print(f"\n[步骤 3] 验证备份分支...")
    code, stdout, stderr = run_command(["git", "fetch", "origin", "--prune"])
    if code != 0:
        print(f"警告: 无法 fetch 验证备份: {stderr}")
    code, stdout, stderr = run_command(["git", "branch", "-r"])
    if f"origin/{backup_branch}" not in stdout.split():
        print(f"警告: 验证失败，备份分支可能未成功推送")
    
    # 清理旧备份分支
    print(f"\n[步骤 4] 检查并清理 30 天以前的旧备份分支...")
    code, stdout, stderr = run_command(["git", "branch", "-r"])
    all_remote_branches = stdout.split()
    
    backup_pattern = re.compile(r"^origin/backup/main/(\d{8})-(\d{6})$")
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    
    backup_branches = []
    old_backups_to_delete = []
    current_backups = []
    
    for branch in all_remote_branches:
        match = backup_pattern.match(branch)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            try:
                branch_datetime = datetime.datetime.strptime(f"{date_str}-{time_str}", "%Y%m%d-%H%M%S")
                backup_branches.append((branch, branch_datetime))
                
                if branch_datetime < thirty_days_ago:
                    old_backups_to_delete.append(branch)
                else:
                    current_backups.append(branch)
            except ValueError:
                continue
    
    deleted_branches = []
    if old_backups_to_delete:
        print(f"找到 {len(old_backups_to_delete)} 个 30 天前的旧备份，准备删除:")
        for branch in old_backups_to_delete:
            print(f"  - {branch}")
            # 删除远程分支
            short_branch_name = branch.replace("origin/", "")
            code, stdout, stderr = run_command(["git", "push", "origin", "--delete", short_branch_name])
            if code == 0:
                deleted_branches.append(branch)
                print(f"  ✓ 已删除: {branch}")
            else:
                print(f"  ✗ 删除失败: {branch}, 错误: {stderr}")
    else:
        print("没有 30 天前的旧备份需要清理")
    
    # 最后再 fetch 一次，更新状态
    run_command(["git", "fetch", "origin", "--prune"], check=False)
    
    # 结果汇报
    print("\n" + "=" * 80)
    print("任务完成报告")
    print("=" * 80)
    print(f"1. 本次创建的备份分支: {backup_branch}")
    print(f"2. 备份对应的 commit hash: {main_commit_hash}")
    print(f"3. 本次删除的旧备份列表:")
    if deleted_branches:
        for b in deleted_branches:
            print(f"   - {b}")
    else:
        print("   无")
    print(f"4. 当前仍保留的备份分支列表:")
    code, stdout, stderr = run_command(["git", "branch", "-r"])
    all_branches_after = stdout.split()
    remaining_backups = []
    for branch in all_branches_after:
        if backup_pattern.match(branch):
            remaining_backups.append(branch)
    if remaining_backups:
        for b in sorted(remaining_backups):
            print(f"   - {b}")
    else:
        print("   无")
    print("=" * 80)


if __name__ == "__main__":
    main()
