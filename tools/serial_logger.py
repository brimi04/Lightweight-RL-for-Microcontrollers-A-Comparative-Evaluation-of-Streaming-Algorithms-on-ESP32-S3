import argparse
import csv
import os
import sys

import serial

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "research", "results", "raw")

FIELDS = ["agent", "env", "seed", "episode", "reward", "steps",
          "memory_bytes", "flops_per_update", "wall_time_s"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="e.g. COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out", default=None, help="Output CSV (default: results/raw/esp32_<agent>_<env>.csv)")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=5)
    rows = []
    header_seen = False

    print(f"Listening on {args.port} @ {args.baud}... (Ctrl+C to stop)")
    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if line == "DONE":
                print("ESP32 reported the end of training.")
                break
            if line.startswith("agent,env,seed"):
                header_seen = True
                continue
            if not header_seen:
                continue

            parts = line.split(",")
            if len(parts) != len(FIELDS):
                print(f"linea ignorada (formato inesperado): {line}")
                continue

            row = dict(zip(FIELDS, parts))
            rows.append(row)
            print(line)
    except KeyboardInterrupt:
        print("\nStopped by the user.")
    finally:
        ser.close()

    if not rows:
        print("No rows received, no CSV saved.")
        return

    out_path = args.out
    if out_path is None:
        agent = rows[0]["agent"]
        env = rows[0]["env"]
        out_path = os.path.join(RESULTS_DIR, f"esp32_{agent}_{env}.csv")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
