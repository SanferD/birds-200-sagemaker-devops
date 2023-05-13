import os


os.system("apt-get update && apt-get install ffmpeg libsm6 libxext6  -y")
os.system("pip install -r /opt/ml/processing/input/requirements.txt")
os.system("python3 /opt/ml/processing/input/processing.py")