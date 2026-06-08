#!/usr/bin/env python3
import subprocess
import re
from datetime import datetime, timedelta
import sys


def run_git_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout.strip(), e.stderr.strip()


def main():
    print("=" * 80)
    print("开始执行主分支定期备份与旧备份清理任务")
    print("=" * 80)
    print()

    # 一、准备与检查
    print("阶段一：准备与检查")
    print("-" * 80)

    # 1. 确认当前目录是 Git 仓库
    code, stdout, stderr = run_git_command("git rev-parse --is-inside-work-tree")
    if code != 0 or stdout != "true":
        print("错误：当前目录不是 Git 仓库")
        return 1

    # 2. 检查是否存在 origin
    code, stdout, stderr = run_git_command("git remote -v")
    if code != 0:
        print("错误：无法获取远程仓库信息")
        return 1

    if "origin" not in stdout:
        print("错误：未找到名为 origin 的远程仓库")
        return 1

    print("✓ 确认远程仓库 origin 存在")

    # 3. 拉取远程分支最新状态
    print("正在执行 git fetch origin --prune...")
    code, stdout, stderr = run_git_command("git fetch origin --prune")
    if code != 0:
        print(f"错误：git fetch 失败\n{stderr}")
        return 1

    print("✓ 拉取远程分支最新状态完成")

    # 4-6. 检查 origin/main 或 origin/master 是否存在
    main_branch = None
    for branch in ["main", "master"]:
        code, stdout, stderr = run_git_command(f"git rev-parse --verify origin/{branch} 2>/dev/null")
        if code == 0:
            main_branch = branch
            break

    if not main_branch:
        print("错误：未找到 origin/main 或 origin/master 分支")
        return 1

    print(f"✓ 使用主分支：origin/{main_branch}")
    print()

    # 二、创建备份分支
    print("阶段二：创建备份分支")
    print("-" * 80)

    # 获取当前时间戳
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    backup_branch_name = f"backup/main/{timestamp}"

    # 获取 origin/main 的最新 commit
    code, commit_hash, stderr = run_git_command(f"git rev-parse origin/{main_branch}")
    if code != 0:
        print(f"错误：无法获取 origin/{main_branch} 的 commit hash\n{stderr}")
        return 1

    print(f"✓ 获取主分支 commit hash: {commit_hash}")

    # 创建备份分支
    print(f"正在创建备份分支 {backup_branch_name}...")
    code, stdout, stderr = run_git_command(f"git push origin origin/{main_branch}:refs/heads/{backup_branch_name}")
    if code != 0:
        print(f"错误：无法推送备份分支\n{stderr}")
        return 1

    print("✓ 备份分支推送成功")

    # 验证备份分支是否存在
    print("正在验证备份分支...")
    code, ls_remote_output, stderr = run_git_command("git ls-remote --heads origin")
    if code != 0:
        print(f"错误：无法获取远程分支列表\n{stderr}")
        return 1

    backup_branch_ref = f"refs/heads/{backup_branch_name}"
    if backup_branch_ref not in ls_remote_output:
        print("错误：备份分支验证失败")
        return 1

    print(f"✓ 备份分支验证成功")
    print(f"  - 分支名：{backup_branch_name}")
    print(f"  - Commit hash：{commit_hash}")
    print(f"  - 备份时间：{now.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 三、清理旧备份分支
    print("阶段三：清理旧备份分支")
    print("-" * 80)

    # 获取所有远程分支
    code, ls_remote_output, stderr = run_git_command("git ls-remote --heads origin")
    if code != 0:
        print(f"警告：无法获取远程分支列表，跳过清理")
        deleted_branches = []
    else:
        # 解析日期的正则表达式
        backup_pattern = re.compile(r"^refs/heads/backup/main/(\d{8})-(\d{6})$")
        thirty_days_ago = now - timedelta(days=30)

        deleted_branches = []
        kept_branches = []

        lines = [line.strip() for line in ls_remote_output.split("\n") if line.strip()]
        print(f"找到 {len(lines)} 个远程分支")

        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                ref = parts[1]
                match = backup_pattern.match(ref)
                if match:
                    date_str = match.group(1)
                    try:
                        branch_date = datetime.strptime(date_str, "%Y%m%d")
                        if branch_date < thirty_days_ago:
                            branch_name = ref.replace("refs/heads/", "")
                            deleted_branches.append(branch_name)
                        else:
                            branch_name = ref.replace("refs/heads/", "")
                            kept_branches.append(branch_name)
                    except ValueError:
                        # 日期格式不对，保留
                        pass

        print(f"找到 {len(deleted_branches)} 个超过 30 天的备份分支需要删除")

        # 删除旧备份
        for branch_name in deleted_branches:
            print(f"正在删除 {branch_name}...")
            code, stdout, stderr = run_git_command(f"git push origin --delete {branch_name}")
            if code != 0:
                print(f"警告：删除 {branch_name} 失败\n{stderr}")
            else:
                print(f"✓ 已删除 {branch_name}")

    print()

    # 四、结果汇报
    print("=" * 80)
    print("结果汇报")
    print("=" * 80)
    print()
    print(f"1. 本次创建的备份分支名称：")
    print(f"   {backup_branch_name}")
    print()
    print(f"2. 备份对应的 commit hash：")
    print(f"   {commit_hash}")
    print()
    print(f"3. 本次删除的旧备份分支列表：")
    if deleted_branches:
        for branch in deleted_branches:
            print(f"   - {branch}")
    else:
        print("   (无)")
    print()
    print(f"4. 当前仍保留的备份分支列表：")
    # 重新获取最新的备份分支
    code, ls_remote_output, stderr = run_git_command("git ls-remote --heads origin")
    if code == 0:
        backup_pattern = re.compile(r"^refs/heads/backup/main/(\d{8})-(\d{6})$")
        current_kept = []
        lines = [line.strip() for line in ls_remote_output.split("\n") if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                ref = parts[1]
                match = backup_pattern.match(ref)
                if match:
                    branch_name = ref.replace("refs/heads/", "")
                    current_kept.append(branch_name)
        
        if current_kept:
            for branch in sorted(current_kept):
                print(f"   - {branch}")
        else:
            print("   (无)")
    else:
        print("   (无法获取)")
    print()
    print("=" * 80)
    print("任务完成")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
