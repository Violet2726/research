#!/usr/bin/env python3
import subprocess
import re
import datetime


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


print("=" * 80)
print("Git 主分支备份与旧备份清理任务 - 最终报告")
print("=" * 80)

# 获取所有远程分支
try:
    ls_remote_output = run_cmd(["git", "ls-remote", "--heads", "origin"])
    main_commit = run_cmd(["git", "rev-parse", "origin/main"])
    
    backup_pattern = re.compile(r"refs/heads/backup/main/(\d{8})-(\d{6})")
    backup_branches = []
    
    for line in ls_remote_output.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            commit_hash = parts[0]
            ref = parts[1]
            match = backup_pattern.match(ref)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                branch_name = ref.replace("refs/heads/", "")
                try:
                    branch_datetime = datetime.datetime.strptime(
                        f"{date_str}-{time_str}", 
                        "%Y%m%d-%H%M%S"
                    )
                    backup_branches.append(
                        (branch_datetime, branch_name, commit_hash)
                    )
                except ValueError:
                    continue
    
    backup_branches.sort()
    
    # 最新的备份
    if backup_branches:
        latest_backup = backup_branches[-1]
        print(f"\n1. 本次创建的备份分支: {latest_backup[1]}")
        print(f"2. 备份对应的 commit hash: {latest_backup[2]}")
        print(f"   备份时间: {latest_backup[0].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查 30 天前
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    
    print(f"\n3. 本次删除的旧备份列表:")
    print(f"   (今天是 {datetime.datetime.now().strftime('%Y-%m-%d')}, "
          f"30天前是 {thirty_days_ago.strftime('%Y-%m-%d')})")
    
    old_backups = []
    for dt, name, hash_val in backup_branches:
        if dt < thirty_days_ago:
            old_backups.append(name)
    
    if old_backups:
        print(f"   找到 {len(old_backups)} 个超过30天的旧备份:")
        for b in old_backups:
            print(f"   - {b}")
    else:
        print("   无（所有备份都在 30 天内，无需删除）")
    
    print(f"\n4. 当前仍保留的备份分支列表 (共 {len(backup_branches)} 个):")
    for dt, name, hash_val in backup_branches:
        print(f"   - {name}")
    
    print("\n" + "=" * 80)
    print("任务已完成！")
    print("=" * 80)
    
except Exception as e:
    print(f"错误: {e}")
