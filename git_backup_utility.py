#!/usr/bin/env python3
import subprocess
import re
from datetime import datetime, timedelta
import sys


def run_command(cmd, capture_output=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            cwd="/workspace"
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def main():
    print("=== Git 主分支定期备份与旧备份清理任务 ===\n")

    # 一、准备与检查
    print("步骤一：准备与检查")
    print("-" * 50)

    # 1. 确认当前目录是 Git 仓库 - 已在外部检查过

    # 2. 执行 git remote -v，确认存在 origin
    print("检查远程仓库 origin ...")
    rc, stdout, stderr = run_command("git remote -v")
    if rc != 0 or "origin" not in stdout:
        print("❌ 错误：找不到远程仓库 origin")
        return 1
    print("✅ 远程仓库 origin 存在")

    # 3. 执行 git fetch origin --prune，拉取远程分支最新状态
    print("\n正在拉取远程分支最新状态 ...")
    rc, stdout, stderr = run_command("git fetch origin --prune")
    if rc != 0:
        print(f"❌ 错误：git fetch 失败 - {stderr}")
        return 1
    print("✅ 成功更新远程分支状态")

    # 4. 检查远程主分支 origin/main 是否存在
    # 5. 如果 origin/main 不存在，则检查 origin/master 是否存在
    print("\n检查主分支 ...")
    rc, stdout, stderr = run_command("git ls-remote --heads origin main")
    main_exists = rc == 0 and len(stdout.strip()) > 0

    rc, stdout, stderr = run_command("git ls-remote --heads origin master")
    master_exists = rc == 0 and len(stdout.strip()) > 0

    target_branch = None
    if main_exists:
        target_branch = "origin/main"
        print(f"✅ 找到主分支 origin/main")
    elif master_exists:
        target_branch = "origin/master"
        print(f"✅ 找到主分支 origin/master")
    else:
        print("❌ 错误：找不到 origin/main 或 origin/master")
        return 1

    # 二、创建备份分支
    print("\n步骤二：创建备份分支")
    print("-" * 50)

    # 2. 生成当前时间戳，格式为 YYYYMMDD-HHMMSS
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    print(f"当前时间戳: {timestamp}")

    # 3. 创建备份分支名
    backup_branch = f"backup/main/{timestamp}"
    print(f"备份分支名: {backup_branch}")

    # 1. 以远程主分支为备份源，获取最新 commit hash
    rc, stdout, stderr = run_command(f"git rev-parse {target_branch}")
    if rc != 0:
        print(f"❌ 错误：无法获取 {target_branch} 的 commit hash - {stderr}")
        return 1
    commit_hash = stdout.strip()
    print(f"备份源 {target_branch} 的 commit hash: {commit_hash}")

    # 4. 创建备份分支并推送到远程
    print("\n正在创建并推送备份分支 ...")
    rc, stdout, stderr = run_command(f"git push origin {commit_hash}:refs/heads/{backup_branch}")
    if rc != 0:
        print(f"❌ 错误：推送备份分支失败 - {stderr}")
        return 1
    print(f"✅ 成功推送备份分支 {backup_branch}")

    # 6. 验证远程备份分支是否存在
    rc, stdout, stderr = run_command(f"git ls-remote --heads origin {backup_branch}")
    if rc != 0 or len(stdout.strip()) == 0:
        print(f"❌ 错误：验证备份分支失败")
        return 1
    print(f"✅ 验证备份分支存在")

    # 三、清理旧备份分支
    print("\n步骤三：清理旧备份分支")
    print("-" * 50)

    # 1. 获取所有远程分支
    rc, stdout, stderr = run_command("git ls-remote --heads origin")
    if rc != 0:
        print(f"❌ 错误：获取远程分支列表失败 - {stderr}")
        return 1

    # 2. 只筛选符合格式的远程备份分支：origin/backup/main/YYYYMMDD-HHMMSS
    backup_pattern = re.compile(r"^refs/heads/(backup/main/(\d{8})-(\d{6}))$")
    backup_branches = []

    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            _, ref = parts
            match = backup_pattern.match(ref)
            if match:
                branch_name = match.group(1)
                date_str = match.group(2)
                time_str = match.group(3)
                try:
                    branch_date = datetime.strptime(date_str, "%Y%m%d")
                    backup_branches.append((branch_name, branch_date))
                except ValueError:
                    continue

    print(f"找到 {len(backup_branches)} 个备份分支")

    # 3-8. 处理旧备份分支
    now = datetime.now()
    cutoff_date = now - timedelta(days=30)
    branches_to_delete = []
    branches_to_keep = []

    for branch_name, branch_date in backup_branches:
        if branch_date < cutoff_date:
            branches_to_delete.append(branch_name)
        else:
            branches_to_keep.append(branch_name)

    print(f"保留 {len(branches_to_keep)} 个最近30天内的备份")
    print(f"准备删除 {len(branches_to_delete)} 个30天前的备份")

    deleted_branches = []
    for branch in branches_to_delete:
        # 再次确认分支名匹配
        if not re.match(r"^backup/main/\d{8}-\d{6}$", branch):
            print(f"⚠️ 跳过不符合格式的分支: {branch}")
            continue
        print(f"正在删除远程分支: {branch}")
        rc, stdout, stderr = run_command(f"git push origin --delete {branch}")
        if rc == 0:
            print(f"✅ 已删除: {branch}")
            deleted_branches.append(branch)
        else:
            print(f"❌ 删除失败: {branch} - {stderr}")

    # 8. 更新本地远程分支缓存
    print("\n更新本地远程分支缓存 ...")
    rc, stdout, stderr = run_command("git fetch origin --prune")
    if rc != 0:
        print(f"⚠️ 警告：git fetch 失败 - {stderr}")

    # 四、结果汇报
    print("\n" + "=" * 50)
    print("=== 结果汇报 ===")
    print("=" * 50)
    print(f"1. 本次创建的备份分支名称: {backup_branch}")
    print(f"2. 备份对应的 commit hash: {commit_hash}")
    print(f"3. 本次删除的旧备份分支列表 (共 {len(deleted_branches)} 个):")
    if deleted_branches:
        for b in deleted_branches:
            print(f"   - {b}")
    else:
        print("   (无)")
    print(f"4. 当前仍保留的备份分支列表 (共 {len(branches_to_keep) - (1 if backup_branch in branches_to_keep else 0)} + 1 个新备份):")
    for b in sorted(branches_to_keep):
        marker = " [NEW]" if b == backup_branch else ""
        print(f"   - {b}{marker}")
    print("\n✅ 任务完成!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
