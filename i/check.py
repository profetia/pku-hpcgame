import sys
import numpy as np

THRESHOLD = 0.0001

def main():
    if len(sys.argv) < 3:
        print("Usage: check.py <ref> <img>")

    ref = np.fromfile(sys.argv[1], dtype=np.float64)
    img = np.fromfile(sys.argv[2], dtype=np.float64)

    delta = np.sqrt(((img - ref) ** 2).sum() / (ref ** 2).sum())

    print("Inaccuracy: ", delta)
    if delta < THRESHOLD:
        print("Ok")
    else:
        print("Fail")

if __name__ == "__main__":
    main()
