import subprocess, sys
for s in ("clustering.py", "evaluation.py"):
    subprocess.run([sys.executable, s], check=True)
