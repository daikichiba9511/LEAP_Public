import argparse
import subprocess


def run_cmd(cmd: list[str]) -> None:
    _cmd = " ".join(cmd)
    print(f"Running: {_cmd}")
    subprocess.run(cmd, check=True)


# ===================================
# main
# ===================================
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

dbg_flag = "--debug" if args.debug else ""

exp_configs = [
    ("exp078", None),
    ("exp080", None),
    ("exp081", None),
    ("exp082", None),
    ("exp084", None),
    ("exp087", None),
]

for exp_ver, train_folds in exp_configs:
    print("Scheduled:", exp_ver, train_folds)


for exp_ver, _train_cfg in exp_configs:
    train_cmd = ["python", "-O", "-m", f"src.exp.{exp_ver}.train"]
    if args.debug:
        train_cmd.append("--debug")
    try:
        run_cmd(train_cmd)
        run_cmd(["python", "-O", "-m", f"src.exp.{exp_ver}.test", "|", "tee", "-a", f"./output/{exp_ver}/infer.log"])
    except Exception as e:
        print(f"Failed: {exp_ver}")
        print(e)
        print()
        continue
