import argparse
import itertools
import json
import subprocess

from typing import Iterator


PAINTER = "painter"


def subprocess_run(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


def painter_job_get() -> dict[str, str]:
    cmd = [PAINTER, "job", "get"]

    output = subprocess_run(cmd)
    return json.loads(output)


VANITY = "vanity"


def run_vanity(prefixs: list[str]) -> list[str]:
    with open(f"{VANITY}.in", "w") as f:
        f.write("\n".join(prefixs))

    cmd = [f"./{VANITY}"]
    subprocess_run(cmd)

    with open(f"{VANITY}.out", "r") as f:
        # read line take only odd line, remove \n
        return [line.strip("") for line in f.readlines()[1::2]]


LIFETIME = 10


def painter_job_submit(keys: dict[str, str]) -> str:
    cmd = [
        PAINTER,
        "job",
        "submit",
        "--r",
        keys["r"],
        "--g",
        keys["g"],
        "--b",
        keys["b"],
        "--jobid",
        keys["jobid"],
    ]

    output = subprocess_run(cmd)
    return json.loads(output)["token"]


def painter_pixel_set(x: int, y: int, token: str) -> None:
    cmd = [
        PAINTER,
        "pixel",
        "set",
        "--x",
        str(x),
        "--y",
        str(y),
        "--token",
        token,
    ]
    subprocess_run(cmd)


def chunk(it: Iterator, size: int) -> Iterator[list]:
    while chunk := list(itertools.islice(it, size)):
        yield chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=str)
    parser.add_argument("--x-offset", type=int)
    parser.add_argument("--y-offset", type=int)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    with open(args.work, "r") as f:
        work: dict[str, list[tuple[int, int]]] = json.load(f)

    x_offset = args.x_offset
    y_offset = args.y_offset

    for color, points in work.items():
        color = "{:02x}".format(int(color))
        for idx, task in enumerate(chunk(points, LIFETIME)):
            job = painter_job_get()
            print(f"Fetched {job['jobid']} for {color}:{idx}")

            prefixs = [
                f'{color}{job["r"]}',
                f'{color}{job["g"]}',
                f'{color}{job["b"]}',
            ]
            keys = run_vanity(prefixs)
            print(f"Generated keys for {color}:{idx}")

            submission = {
                "r": keys[0],
                "g": keys[1],
                "b": keys[2],
                "jobid": job["jobid"],
            }
            token = painter_job_submit(submission)
            print(f"Submited job for {color}:{idx}")

            for x, y in task:
                painter_pixel_set(x + x_offset, y + y_offset, token)

            print(f"Painted {len(task)} pixels for {color}:{idx}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
