#!/usr/bin/env python3
import subprocess
import re
from datetime import datetime, timedelta
import sys


def run_git_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip(), e.stderr.strip(), e.returncode


def is_git_repo():
    _, _, code = run_git_command("git rev-parse --is-inside-work-tree")
    return code == 0


def get_remote_backup_branches():
    # 不使用 --heads 选项，直接获取所有 refs 然后过滤
    ls_remote_output, _, _ = run_git_command("git ls-remote origin")
    backup_pattern = re.compile(r"refs/heads/(backup/main/(\d{8})-(\d{6}))")
    branches = []
    for line in ls_remote_output.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = backup_pattern.search(line)
        if match:
            full_ref, date_str, time_str = match.groups()
            try:
                branch_date = datetime.strptime(f"{date_str}-{time_str}", "%Y%m%d-%H%M%S")
                branches.append((full_ref, branch_date))
            except ValueError:
                continue
    return branches


def main():
    backup_created = False
    backup_branch = None
    commit_hash = None
    deleted_branches = []

    # 一、准备与检查
    print("=== 一、准备与检查 ===")

    if not is_git_repo():
        print("错误：当前目录不是 Git 仓库")
        sys.exit(1)
    print("✓ 当前是 Git 仓库")

    remote_output, _, _ = run_git_command("git remote -v")
    if "origin" not in remote_output:
        print("错误：未找到 origin 远程仓库")
        sys.exit(1)
    print("✓ 存在 origin 远程仓库")

    print("正在拉取远程分支最新状态...")
    run_git_command("git fetch origin --prune")

    main_branch = None
    branches_output, _, _ = run_git_command("git branch -r")
    if "origin/main" in branches_output:
        main_branch = "origin/main"
    elif "origin/master" in branches_output:
        main_branch = "origin/master"
    else:
        print("错误：未找到 origin/main 或 origin/master 分支")
        sys.exit(1)
    print(f"✓ 使用主分支: {main_branch}")

    # 获取主分支最新 commit hash
    commit_hash, _, _ = run_git_command(f"git rev-parse {main_branch}")
    print(f"✓ 主分支最新 commit: {commit_hash}")

    # 二、创建备份分支
    print("\n=== 二、创建备份分支 ===")
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    backup_branch = f"backup/main/{timestamp}"
    print(f"创建备份分支: {backup_branch}")

    # 推送备份分支到 origin
    stdout, stderr, code = run_git_command(f"git push origin {commit_hash}:refs/heads/{backup_branch}")
    if code == 0:
        # 使用 git ls-remote 验证备份分支是否存在
        remote_branches = get_remote_backup_branches()
        if any(ref == backup_branch for ref, _ in remote_branches):
            print("✓ 备份分支创建成功")
            backup_created = True
        else:
            print("警告：推送成功但验证失败")
    else:
        print(f"警告：推送备份分支失败")
        print(f"  标准输出: {stdout}")
        print(f"  错误输出: {stderr}")

    # 三、清理旧备份分支
    print("\n=== 三、清理旧备份分支 ===")
    thirty_days_ago = datetime.now() - timedelta(days=30)
    remote_branches = get_remote_backup_branches()
    all_backup_branches = [ref for ref, _ in remote_branches]
    branches_to_delete = [
        ref for ref, date in remote_branches if date < thirty_days_ago
    ]

    print(f"找到 {len(all_backup_branches)} 个备份分支")
    print(f"其中 {len(branches_to_delete)} 个超过 30 天，将被删除")

    for branch in branches_to_delete:
        print(f"尝试删除分支: {branch}")
        stdout, stderr, code = run_git_command(f"git push origin --delete {branch}")
        if code == 0:
            deleted_branches.append(branch)
            print(f"✓ 删除成功: {branch}")
        else:
            print(f"✗ 删除失败: {branch}")
            print(f"  错误输出: {stderr}")

    # 更新远程分支缓存
    run_git_command("git fetch origin --prune")

    # 四、结果汇报
    print("\n=== 四、结果汇报 ===")
    print(f"1. 本次创建的备份分支: {backup_branch if backup_created else '创建失败'}")
    print(f"2. 备份 commit hash: {commit_hash}")
    print(f"3. 本次删除的旧备份分支: {', '.join(deleted_branches) if deleted_branches else '无'}")
    print("4. 当前保留的备份分支:")
    remaining_branches = get_remote_backup_branches()
    for ref, _ in remaining_branches:
        print(f"   - origin/{ref}")


if __name__ == "__main__":
    main()
