import os

def generate_markdown_report(run_dir: str, title: str, content: str):
    path = os.path.join(run_dir, "reports", "report.md")
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(content)
