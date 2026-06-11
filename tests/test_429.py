"""测试 MIMO API 429 限流"""

import os
import time

import requests

API_KEY = os.getenv("XIAOMI_MIMO_API_KEY")
BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MODEL = "mimo-v2.5"

RPM = 20  # 每分钟请求数
DURATION_MIN = 2  # 持续分钟数
INTERVAL = 60.0 / RPM  # 每个请求间隔秒数
TOTAL = RPM * DURATION_MIN

# 直连不走代理（代理会干扰 SSL 握手）
PROXIES = {"https": None, "http": None}


# 创建 Session 并禁用环境代理
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.proxies = PROXIES


def send_request(index: int) -> dict:
    """发送单个 chat completion 请求，返回状态信息。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"说一个数字：{index}"}],

    }
    start = time.time()
    try:
        resp = SESSION.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        elapsed = time.time() - start
        return {
            "index": index,
            "status": resp.status_code,
            "elapsed": round(elapsed, 3),
            "body": resp.text[:200] if resp.status_code != 200 else "ok",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "index": index,
            "status": "error",
            "elapsed": round(elapsed, 3),
            "body": f"{type(e).__name__}: {e}"[:200],
        }


def main():
    print(f"开始测试：{RPM} RPM × {DURATION_MIN} min = {TOTAL} 请求，间隔 {INTERVAL:.3f}s")
    print(f"模型：{MODEL}")
    print(f"端点：{BASE_URL}")
    print("代理：已禁用（直连）")
    print("-" * 60)

    stats = {"total": 0, "ok": 0, "429": 0, "other_err": 0}

    for i in range(1, TOTAL + 1):
        result = send_request(i)
        stats["total"] += 1

        status = result["status"]
        if status == 200:
            stats["ok"] += 1
            tag = "✓"
        elif status == 429:
            stats["429"] += 1
            tag = "✗ 429"
        else:
            stats["other_err"] += 1
            tag = f"✗ {status}"

        print(
            f"[{i:>3}/{TOTAL}] {tag:>6}  "
            f"{result['elapsed']:.3f}s  {result['body'][:80]}"
        )

        if i < TOTAL:
            time.sleep(INTERVAL)

    print("-" * 60)
    print(
        f"完成。总计 {stats['total']}，"
        f"成功 {stats['ok']}，"
        f"429 {stats['429']}，"
        f"其他错误 {stats['other_err']}"
    )


if __name__ == "__main__":
    main()
