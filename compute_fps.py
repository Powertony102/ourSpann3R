# save as calc_fps_avg.py
import re
import sys

def calc_avg_fps_from_stream(stream):
    fps_values = []
    pattern = re.compile(r"FPS_avg:\s*([0-9.]+)")

    for line in stream:
        m = pattern.search(line)
        if m:
            fps_values.append(float(m.group(1)))

    if not fps_values:
        print("No FPS_avg found.")
        return

    avg = sum(fps_values) / len(fps_values)
    print(f"Average FPS: {avg:.3f}")

if __name__ == "__main__":
    # 用法1：python calc_fps_avg.py log.txt
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            calc_avg_fps_from_stream(f)
    else:
        # 用法2：直接管道输入：python calc_fps_avg.py < log.txt
        calc_avg_fps_from_stream(sys.stdin)
