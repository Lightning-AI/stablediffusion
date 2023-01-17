from subprocess import PIPE, Popen

# Credits to AITemplate Team
# https://github.com/facebookincubator/AITemplate/blob/main/python/aitemplate/testing/detect_target.py
def _detect_cuda():
    try:
        proc = Popen(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, _ = proc.communicate()
        stdout = stdout.decode("utf-8")
        if "A100" in stdout or "RTX 30" in stdout or "A30" in stdout or "A10" in stdout:
            return "80"
        if "V100" in stdout:
            return "70"
        if "T4" in stdout:
            return "75"
        return None
    except Exception:
        return None