#!/usr/bin/env python3
import datetime
import subprocess
import sys
import re

def run_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd="/workspace"
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def main():
    created_backup = None
    deleted_branches = []
    kept_branches = []

    # 获取当前时间
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    backup_branch = f"backup/main/{timestamp}"

    # 1. 获取 origin/main 的 commit hash
    print("正在获取 origin/main 的最新 commit...")
    code, main_commit, err = run_command("git rev-parse origin/main")
    if code != 0:
        print(f"错误: 无法获取 origin/main 的 commit: {err}")
        return False

    # 2. 创建备份分支并推送
    print(f"正在创建备份分支: {backup_branch}")
    code, out, err = run_command(f"git push origin origin/main:refs/heads/{backup_branch}")
    if code != 0:
        print(f"错误: 无法推送备份分支: {err}")
        return False

    created_backup = {
        "name": backup_branch,
        "commit": main_commit,
        "time": now.strftime("%Y-%m-%d %H:%M:%S")
    }
    print(f"✓ 备份分支已创建: {backup_branch}")

    # 3. 获取所有远程分支
    print("\n正在获取所有远程备份分支...")
    code, branches, err = run_command("git branch -r --format='%(refname:short)'")
    if code != 0:
        print(f"错误: 无法获取远程分支列表: {err}")
        return False

    # 4. 筛选符合 backup/main/YYYYMMDD-HHMMSS 格式的分支
    backup_pattern = re.compile(r"^origin/backup/main/(\d{8})-(\d{6})$")
    backup_branches = []

    for branch in branches.splitlines():
        match = backup_pattern.match(branch.strip())
        if match:
            date_str = match.group(1)
            try:
                branch_date = datetime.datetime.strptime(date_str, "%Y%m%d")
                backup_branches.append({
                    "full_name": branch.strip(),
                    "short_name": branch.strip().replace("origin/", ""),
                    "date": branch_date
                })
            except ValueError:
                continue

    print(f"找到 {len(backup_branches)} 个符合格式的备份分支")

    # 5. 计算 30 天前的日期
    cutoff_date = now - datetime.timedelta(days=30)
    print(f"删除截止日期: {cutoff_date.strftime('%Y-%m-%d')}")

    # 6. 筛选需要删除和保留的分支
    for branch in backup_branches:
        if branch["date"] < cutoff_date:
            deleted_branches.append(branch["short_name"])
        else:
            kept_branches.append(branch["short_name"])
    
    # 确保新创建的备份也在保留列表中（如果符合条件）
    if backup_branch not in kept_branches and backup_branch not in deleted_branches:
        kept_branches.append(backup_branch)

    # 7. 删除旧分支
    if deleted_branches:
        print(f"\n正在删除 {len(deleted_branches)} 个旧备份分支...")
        for branch in deleted_branches:
            code, out, err = run_command(f"git push origin --delete {branch}")
            if code == 0:
                print(f"✓ 已删除: {branch}")
            else:
                print(f"✗ 删除失败 {branch}: {err}")

    # 8. 再次执行 git fetch --prune 更新本地缓存
    print("\n正在更新远程分支缓存...")
    run_command("git fetch origin --prune")

    # 9. 输出结果汇报
    print("\n" + "="*60)
    print("结果汇报")
    print("="*60)

    print("\n1. 本次创建的备份分支:")
    if created_backup:
        print(f"   - 分支名: {created_backup['name']}")
        print(f"   - Commit: {created_backup['commit']}")
        print(f"   - 备份时间: {created_backup['time']}")

    print("\n2. 本次删除的旧备份分支列表:")
    if deleted_branches:
        for branch in deleted_branches:
            print(f"   - {branch}")
    else:
        print("   - 没有删除任何分支")

    print("\n3. 当前仍保留的备份分支列表:")
    if kept_branches:
        for branch in sorted(kept_branches):
            print(f"   - {branch}")
    else:
        print("   - 没有保留的备份分支")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
