#!/usr/bin/env python3
import subprocess
import datetime
import re
import sys

def run_cmd(cmd, cwd=None):
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()

def main():
    # 仓库根目录
    repo_path = "/workspace"
    
    # 1. 准备与检查
    print("=== 准备与检查 ===")
    
    # 检查 origin/main 是否存在
    code, _, _ = run_cmd("git rev-parse --verify origin/main", cwd=repo_path)
    if code != 0:
        code, _, _ = run_cmd("git rev-parse --verify origin/master", cwd=repo_path)
        if code != 0:
            print("错误：origin/main 和 origin/master 都不存在！")
            sys.exit(1)
        main_branch = "master"
    else:
        main_branch = "main"
    
    print(f"使用远程主分支：origin/{main_branch}")
    
    # 获取 origin/main 的最新 commit hash
    code, commit_hash, _ = run_cmd(f"git rev-parse origin/{main_branch}", cwd=repo_path)
    if code != 0:
        print(f"错误：无法获取 origin/{main_branch} 的 commit hash")
        sys.exit(1)
    print(f"当前 origin/{main_branch} 最新 commit：{commit_hash}")
    
    # 2. 创建备份分支
    print("\n=== 创建备份分支 ===")
    
    # 生成时间戳 YYYYMMDD-HHMMSS
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    backup_branch = f"backup/{main_branch}/{timestamp}"
    
    print(f"备份分支名：{backup_branch}")
    
    # 创建备份分支
    code, _, stderr = run_cmd(f"git push origin origin/{main_branch}:refs/heads/{backup_branch}", cwd=repo_path)
    if code != 0:
        print(f"错误：无法推送备份分支 {backup_branch}：{stderr}")
        sys.exit(1)
    
    print(f"成功创建备份分支 {backup_branch}")
    
    # 验证备份分支存在
    print("正在更新远程分支缓存...")
    run_cmd("git fetch origin --prune", cwd=repo_path)
    code, _, _ = run_cmd(f"git rev-parse --verify origin/{backup_branch}", cwd=repo_path)
    if code != 0:
        print(f"警告：备份分支 {backup_branch} 推送到远程后验证失败")
    else:
        print(f"验证成功：备份分支 {backup_branch} 已成功创建在远程")
    
    # 3. 清理旧备份分支
    print("\n=== 清理旧备份分支 ===")
    
    # 获取所有远程分支
    code, remote_branches, _ = run_cmd("git branch -r", cwd=repo_path)
    if code != 0:
        print("错误：无法获取远程分支列表")
        sys.exit(1)
    
    # 分支名格式：origin/backup/main/YYYYMMDD-HHMMSS
    branch_pattern = re.compile(r"^origin/backup/(main|master)/(\d{8})-(\d{6})$")
    
    old_branches_to_delete = []
    all_backup_branches = []
    
    # 计算 30 天前的日期
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
    
    for branch in remote_branches.split('\n'):
        branch = branch.strip()
        if not branch:
            continue
        
        match = branch_pattern.match(branch)
        if match:
            branch_type, date_str, time_str = match.groups()
            # 解析日期：YYYYMMDD
            try:
                branch_date = datetime.datetime.strptime(date_str, "%Y%m%d")
                branch_name = branch.replace("origin/", "")
                all_backup_branches.append(branch_name)
                
                if branch_date < cutoff_date:
                    old_branches_to_delete.append(branch_name)
                    print(f"找到旧备份分支：{branch_name}（日期：{date_str}）")
            except ValueError:
                continue
    
    deleted_branches = []
    
    for branch in old_branches_to_delete:
        print(f"正在删除远程分支：{branch}")
        code, _, stderr = run_cmd(f"git push origin --delete {branch}", cwd=repo_path)
        if code == 0:
            deleted_branches.append(branch)
            print(f"成功删除：{branch}")
        else:
            print(f"删除失败：{branch}，错误：{stderr}")
    
    # 更新远程分支缓存
    print("\n更新远程分支缓存...")
    run_cmd("git fetch origin --prune", cwd=repo_path)
    
    # 重新获取当前备份分支列表
    code, remote_branches, _ = run_cmd("git branch -r", cwd=repo_path)
    current_backup_branches = []
    for branch in remote_branches.split('\n'):
        branch = branch.strip()
        if not branch:
            continue
        match = branch_pattern.match(branch)
        if match:
            current_backup_branches.append(branch.replace("origin/", ""))
    
    # 4. 结果汇报
    print("\n" + "="*50)
    print("结果汇报")
    print("="*50)
    print(f"\n1. 本次创建的备份分支：")
    print(f"   - 分支名称：{backup_branch}")
    print(f"   - 备份 commit：{commit_hash}")
    print(f"   - 备份时间：{now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n2. 本次删除的旧备份分支（{len(deleted_branches)} 个）：")
    if deleted_branches:
        for branch in deleted_branches:
            print(f"   - {branch}")
    else:
        print("   无")
    
    print(f"\n3. 当前仍保留的备份分支（{len(current_backup_branches)} 个）：")
    if current_backup_branches:
        current_backup_branches.sort()
        for branch in current_backup_branches:
            print(f"   - {branch}")
    else:
        print("   无")
    
    print("\n" + "="*50)
    print("任务完成")
    print("="*50)

if __name__ == "__main__":
    main()
