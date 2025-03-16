import subprocess
import time

def cmd(command: list[str], time_fmt=None):
    start = time.time()
    process = subprocess.Popen(command, shell=False,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               encoding='utf-8',
                               bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    end = time.time()

    print(time_fmt.format(end - start))
