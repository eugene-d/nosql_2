import subprocess, html
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<style>
body {{ margin: 0; padding: 0; background: #1e1e1e; }}
.terminal {{
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Ubuntu Mono', 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.4;
    padding: 12px 16px;
    white-space: pre-wrap;
    word-wrap: break-word;
    max-width: 900px;
}}
.prompt {{ color: #4ec9b0; }}
.cmd {{ color: #dcdcaa; }}
</style>
</head>
<body>
<div class="terminal">{content}</div>
</body>
</html>"""


def clean_tqdm(text):
    """collapse tqdm progress bar lines into final one"""
    lines = text.split("\n")
    out = []
    prev_tqdm = False
    for line in lines:
        is_tqdm = ("it/s]" in line or "s/it]" in line) and ("|" in line or "%" in line)
        is_parsing = line.startswith("parsing:")
        if is_tqdm or is_parsing:
            if prev_tqdm:
                out[-1] = line
            else:
                out.append(line)
            prev_tqdm = True
        else:
            out.append(line)
            prev_tqdm = False
    return "\n".join(out)


def colorize(cmd, text):
    text = clean_tqdm(text)
    header = f'<span class="prompt">$ </span><span class="cmd">{html.escape(cmd)}</span>'
    body = html.escape(text)
    return header + "\n" + body


def render(name, content):
    page = TEMPLATE.format(content=content)
    html_path = ROOT / f"{name}.html"
    html_path.write_text(page)
    png_path = ROOT / f"{name}.png"
    subprocess.run([
        "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
        f"--screenshot={png_path}", "--window-size=960,4000",
        str(html_path),
    ], capture_output=True)
    html_path.unlink()

    # crop
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(png_path)
        arr = np.array(img)
        for y in range(arr.shape[0]-1, 0, -1):
            if arr[y, :, :3].std() > 2:
                break
        img.crop((0, 0, img.width, y + 16)).save(png_path)
    except Exception:
        pass
    print(f"  {name}.png OK")


# read saved actual outputs
for name, output_file in [
    ("01_prepare_data", "/tmp/actual_01.txt"),
    ("02_embed", "/tmp/actual_02.txt"),
    ("03_load_to_pinecone", "/tmp/actual_03.txt"),
    ("04_search", "/tmp/actual_04.txt"),
    ("05_chunking", "/tmp/actual_05.txt"),
    ("06_hybrid_search", "/tmp/actual_06.txt"),
]:
    p = Path(output_file)
    if not p.exists():
        print(f"  SKIP {name} -- no output file {output_file}")
        continue
    text = p.read_text()
    content = colorize(f"python scripts/{name}.py", text)
    render(name, content)
