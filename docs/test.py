import os
import subprocess

# Download csv files
current = os.getcwd()
os.chdir(current + "/../data")
if not os.path.isdir("./csv_files"):
    subprocess.call(["sh", "./download.sh"])
os.chdir(current)
