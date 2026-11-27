import os, sys, time, requests

url, dst = sys.argv[1], sys.argv[2]
pos = os.path.getsize(dst) if os.path.exists(dst) else 0
t0 = time.time()
last_report = pos
with open(dst, "ab") as f:
    for attempt in range(20):
        headers = {"User-Agent": "Mozilla/5.0"}
        if pos:
            headers["Range"] = f"bytes={pos}-"
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True)
            if r.status_code == 200 and pos:
                print("server ignored range, restarting", flush=True)
                f.truncate(0)
                pos = 0
            if r.status_code not in (200, 206):
                print("status", r.status_code, flush=True)
                time.sleep(3)
                continue
            for chunk in r.iter_content(8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pos += len(chunk)
                if pos - last_report > 200 * 1024 * 1024:
                    print(f"{pos/1e9:.2f} GB  {pos/max(time.time()-t0,1)/1e6:.1f} MB/s", flush=True)
                    last_report = pos
            cr = r.headers.get("Content-Range", "")
            total = int(cr.split("/")[-1]) if "/" in cr else 0
            if total and pos >= total:
                print("complete", pos, flush=True)
                break
        except Exception as e:
            print("retry", type(e).__name__, str(e)[:80], flush=True)
            time.sleep(2)
print("final bytes", os.path.getsize(dst))
