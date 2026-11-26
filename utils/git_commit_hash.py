import subprocess

commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()