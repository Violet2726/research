#!/usr/bin/env python3
import subprocess
import sys
import re
from datetime import datetime, timedelta

def run_command(cmd, check=True, capture_output=True):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            cwd='/workspace'
        )
        if check and result.returncode != 0:
            print(f"命令执行失败: {cmd}")
            print(f"错误: {result.stderr}")
            return None
        return result
    except Exception as e:
        print(f"执行命令时发生异常: {e}")
        return None

def main():
    print("=" * 60)
    print("Git 主分支备份与旧备份清理任务")
    print("=" * 60)
    print()

    # 一、准备与检查
    print("步骤 1: 准备与检查")
    print("-" * 60)

    # 1. 确认当前目录是 Git 仓库（已通过之前的命令确认）
    # 2. 确认 origin 存在
    remote_check = run_command("git remote -v")
    if remote_check is None or "origin" not in remote_check.stdout:
        print("错误: 未找到 origin 远程仓库")
        return 1
    print("✓ origin 远程仓库存在")

    # 3. 执行 git fetch origin --prune
    print("正在拉取远程分支最新状态...")
    fetch_result = run_command("git fetch origin --prune")
    if fetch_result is None:
        print("错误: git fetch 失败")
        return 1
    print("✓ git fetch 完成")

    # 4. 检查远程主分支
    branches_result = run_command("git ls-remote --heads origin")
    if branches_result is None:
        print("错误: 获取远程分支列表失败")
        return 1

    remote_branches = branches_result.stdout
    main_branch = None
    if "refs/heads/main" in remote_branches:
        main_branch = "main"
    elif "refs/heads/master" in remote_branches:
        main_branch = "master"

    if main_branch is None:
        print("错误: 未找到 origin/main 或 origin/master 分支")
        return 1
    print(f"✓ 找到主分支: origin/{main_branch}")
    print()

    # 二、创建备份分支
    print("步骤 2: 创建备份分支")
    print("-" * 60)

    # 生成备份分支名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_branch = f"backup/main/{timestamp}"
    print(f"备份分支名称: {backup_branch}")

    # 获取 origin/main 的最新 commit hash
    commit_hash_result = run_command(f"git rev-parse origin/{main_branch}")
    if commit_hash_result is None:
        print("错误: 获取 commit hash 失败")
        return 1
    commit_hash = commit_hash_result.stdout.strip()
    print(f"备份 commit hash: {commit_hash}")

    # 创建并推送备份分支
    print(f"正在推送备份分支...")
    push_result = run_command(f"git push origin origin/{main_branch}:refs/heads/{backup_branch}")
    if push_result is None:
        print("错误: 推送备份分支失败")
        return 1
    print("✓ 备份分支推送成功")
    print()

    # 验证备份分支存在
    verify_result = run_command(f"git ls-remote origin refs/heads/{backup_branch}")
    if verify_result is None or backup_branch not in verify_result.stdout:
        print("警告: 无法验证备份分支是否存在")
    else:
        print("✓ 备份分支已验证存在")
    print()

    # 三、清理旧备份分支
    print("步骤 3: 清理旧备份分支")
    print("-" * 60)

    # 获取所有远程分支
    all_branches_result = run_command("git ls-remote --heads origin")
    if all_branches_result is None:
        print("错误: 获取远程分支列表失败")
        return 1

    # 定义备份分支的正则表达式
    backup_branch_pattern = re.compile(r"refs/heads/(backup/main/(\d{8})-(\d{6}))")
    
    # 解析日期并筛选旧备份
    now = datetime.now()
    cutoff_date = now - timedelta(days=30)
    branches_to_delete = []
    all_backup_branches = []

    for line in all_branches_result.stdout.splitlines():
        match = backup_branch_pattern.search(line)
        if match:
            branch_full = match.group(1)
            date_str = match.group(2)
            try:
                branch_date = datetime.strptime(date_str, "%Y%m%d")
                all_backup_branches.append(branch_full)
                if branch_date < cutoff_date:
                    branches_to_delete.append(branch_full)
            except ValueError:
                continue

    print(f"找到 {len(all_backup_branches)} 个备份分支")
    print(f"需要删除 {len(branches_to_delete)} 个 30 天前的旧备份")

    # 删除旧备份分支
    deleted_branches = []
    for branch in branches_to_delete:
        print(f"正在删除: {branch}")
        delete_result = run_command(f"git push origin --delete {branch}")
        if delete_result is not None:
            deleted_branches.append(branch)
            print(f"✓ 已删除: {branch}")
        else:
            print(f"✗ 删除失败: {branch}")

    # 再次 fetch 更新本地远程分支缓存
    print("\n正在更新远程分支缓存...")
    run_command("git fetch origin --prune", check=False)
    print()

    # 四、结果汇报
    print("=" * 60)
    print("结果汇报")
    print("=" * 60)
    print()
    print(f"1. 本次创建的备份分支名称: {backup_branch}")
    print(f"2. 备份对应的 commit hash: {commit_hash}")
    print()
    print("3. 本次删除的旧备份分支列表:")
    if deleted_branches:
        for branch in deleted_branches:
            print(f"   - {branch}")
    else:
        print("   (无)")
    print()

    # 获取当前保留的备份分支
    remaining_backups = [b for b in all_backup_branches if b not in deleted_branches]
    # 加上新创建的备份分支
    if backup_branch not in remaining_backups:
        remaining_backups.append(backup_branch)
    remaining_backups.sort()

    print("4. 当前仍保留的备份分支列表:")
    if remaining_backups:
        for branch in remaining_backups:
            print(f"   - {branch}")
    else:
        print("   (无)")
    print()
    print("✓ 任务执行完成")
    return 0

if __name__ == "__main__":
    sys.exit(main())
