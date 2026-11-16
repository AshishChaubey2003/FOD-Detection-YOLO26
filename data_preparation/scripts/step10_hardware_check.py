#!/usr/bin/env python3
"""
Step 10, Section 3: Hardware inspection for YOLO model selection.

READS actual system/hardware info only (read-only). WRITES only:
  outputs/step10_model_selection/hardware_report.txt

Run this with the SAME Python/venv you will train with, e.g. from
D:\\FOD_project with venv active:
    venv\\Scripts\\activate
    python src\\step10_hardware_check.py

Every value below is read directly from the OS/Python/torch/ultralytics at
run time. If something can't be determined, it is reported as "unknown" --
nothing is guessed or invented.
"""
import argparse
import platform
import shutil
import sys
from pathlib import Path


def say(lines, s=""):
    print(s)
    lines.append(s)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    say(lines, "=" * 74)
    say(lines, "STEP 10 - HARDWARE REPORT (read-only inspection, real values only)")
    say(lines, "=" * 74)

    # ---- Python ----
    say(lines, f"\nPython version: {sys.version.replace(chr(10), ' ')}")
    say(lines, f"Python executable: {sys.executable}")

    # ---- CPU ----
    say(lines, "\n--- CPU ---")
    cpu_name = "unknown"
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        else:
            cpu_name = platform.processor() or "unknown"
    except Exception as e:
        say(lines, f"[WARNING] CPU name registry read fail hui: {e} -- platform.processor() try kar rahe hain.")
        cpu_name = platform.processor() or "unknown"
    say(lines, f"CPU model: {cpu_name}")
    say(lines, f"Architecture: {platform.machine()}")

    try:
        import psutil
        say(lines, f"Logical cores: {psutil.cpu_count(logical=True)}")
        say(lines, f"Physical cores: {psutil.cpu_count(logical=False)}")
    except ImportError:
        say(lines, f"Logical cores (os.cpu_count): {__import__('os').cpu_count()}")
        say(lines, "[WARNING] psutil import nahi hui -- physical core count aur RAM details skip ho rahe hain.")
        psutil = None

    # ---- RAM ----
    say(lines, "\n--- RAM ---")
    if psutil is not None:
        vm = psutil.virtual_memory()
        say(lines, f"Total RAM: {vm.total / (1024**3):.2f} GB")
        say(lines, f"Available RAM (is waqt): {vm.available / (1024**3):.2f} GB")
    else:
        say(lines, "unknown (psutil missing)")

    # ---- GPU / CUDA / PyTorch ----
    say(lines, "\n--- GPU / CUDA / PyTorch ---")
    try:
        import torch
        say(lines, f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        say(lines, f"CUDA available: {cuda_available}")
        if cuda_available:
            n = torch.cuda.device_count()
            say(lines, f"CUDA device count: {n}")
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                say(lines, f"  GPU {i}: {name} | VRAM: {vram_gb:.2f} GB | "
                           f"compute capability: {props.major}.{props.minor}")
            say(lines, f"torch.version.cuda (CUDA toolkit version PyTorch built against): {torch.version.cuda}")
        else:
            say(lines, "GPU: NO CUDA GPU DETECTED. Training/inference is CPU-only in this environment "
                        "unless a CUDA-capable GPU + matching PyTorch build is set up.")
            say(lines, "NOTE: this only confirms CUDA (NVIDIA) GPU absence. If this machine has an AMD or "
                        "Intel GPU, standard PyTorch here still won't use it without a different build "
                        "(ROCm for AMD, Intel Extension for PyTorch / OpenVINO for Intel) -- not checked here.")
    except ImportError:
        say(lines, "[ABORT-ish] PyTorch import NAHI hui is environment mein -- yeh venv hai jahan "
                    "'pip install ultralytics' chalaya tha? torch ultralytics ki dependency hai, isliye "
                    "iska missing hona ajeeb hai. Confirm karein ki aap sahi venv mein hain "
                    "(prompt mein '(venv)' dikhna chahiye).")

    # ---- Ultralytics ----
    say(lines, "\n--- Ultralytics ---")
    try:
        import ultralytics
        say(lines, f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        say(lines, "ultralytics import nahi hui is environment mein.")

    # ---- Disk space ----
    say(lines, "\n--- Disk space ---")
    for drive in a.drives:
        try:
            total, used, free = shutil.disk_usage(drive)
            say(lines, f"{drive}  total={total/(1024**3):.1f} GB  used={used/(1024**3):.1f} GB  "
                       f"free={free/(1024**3):.1f} GB")
        except Exception as e:
            say(lines, f"{drive}  [could not read: {e}]")

    say(lines, "\n" + "=" * 74)
    say(lines, "Yeh poori report upar jo dikh rahi hai, exactly wahi hai jo hardware_report.txt mein likhi gayi.")
    say(lines, "=" * 74)

    report_path = out / "hardware_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{report_path} likhi gayi.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/step10_model_selection")
    ap.add_argument("--drives", nargs="+", default=["D:\\"])
    main(ap.parse_args())