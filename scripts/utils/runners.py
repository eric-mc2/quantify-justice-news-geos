import subprocess
import time
import re

def cmd(command: str | list[str], time_fmt=None, env=None):
    if isinstance(command, str):
        command = re.sub(r"\s+", " ", command).split(" ")
    
    start = time.time()
    process = subprocess.Popen(command, 
                               shell=False, 
                               env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               encoding='utf-8',
                               bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    status = process.wait()
    end = time.time()

    if time_fmt:
        print(time_fmt.format(end - start))

    if status:
        raise ChildProcessError(f"Exited with {status} from command: {command}")