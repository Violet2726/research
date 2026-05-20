#!/usr/bin/env python3
import datetime
import subprocess
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
    # 获取远程分支列表
    print("正在获取所有远程分支...")
    code, branches, err = run_command("git ls-remote origin")
    if code != 0:
        print(f"错误: {err}")
        return

    # 筛选备份分支
    backup_pattern = re.compile(r"^[0-9a-f]+\s+refs/heads/(backup/main/(\d{8})-(\d{6}))$")
    backup_branches = []

    for line in branches.splitlines():
        match = backup_pattern.match(line.strip())
        if match:
            branch_name = match.group(1)
            date_str = match.group(2)
            time_str = match.group(3)
            try:
                branch_date = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
                backup_branches.append({
                    "name": branch_name,
                    "date": branch_date
                })
            except ValueError:
                continue

    # 按日期排序
    backup_branches.sort(key=lambda x: x["date"], reverse=True)

    print("\n" + "="*60)
    print("当前备份分支列表")
    print("="*60)
    
    if backup_branches:
        for i, branch in enumerate(backup_branches):
            print(f"{i+1}. {branch['name']}")
            print(f"   创建时间: {branch['date'].strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    else:
        print("没有找到备份分支")

if __name__ == "__main__":
    main()
