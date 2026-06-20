import re
import requests
import time

# 从 references.bib 中提取所有 DOI 和对应的 citation key
with open("references.bib", "r", encoding="utf-8") as f:
    content = f.read()

# 匹配每个条目和其中的 doi
entries = re.split(r'\n@\w+{', content)
for entry in entries[1:]:  # 跳过文件头
    # 提取 citation key
    key_match = re.match(r'([^,]+),', entry)
    if not key_match:
        continue
    key = key_match.group(1).strip()

    # 提取 DOI
    doi_match = re.search(r'doi\s*=\s*\{([^}]+)\}', entry)
    if not doi_match:
        continue
    doi = doi_match.group(1).strip()

    # 提取标题（用于核对）
    title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry)
    if title_match:
        bib_title = title_match.group(1)[:100]
    else:
        bib_title = "N/A"

    # 查询 CrossRef
    try:
        url = f"https://api.crossref.org/works/{doi}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            real_title = data['message']['title'][0][:100]
            print(f"✅ {key}: {doi}")
            print(f"   BibTeX: {bib_title}")
            print(f"   CrossRef: {real_title}")
            print()
        else:
            print(f"⚠️ {key}: {doi} -> HTTP {resp.status_code}")
    except Exception as e:
        print(f"❌ {key}: {doi} -> Error: {e}")

    time.sleep(1)