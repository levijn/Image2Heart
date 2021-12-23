from os import walk
import shutil
from pathlib import Path

root_path = "C:\\Users\\levij\\TU Delft\\Engineering with AI\\Capstone\\DataACDC\\simpledata"

f = []
for (dirpath, dirnames, filenames) in walk(root_path):
    f.extend(dirnames)
    break

for folder in f:
    path_f = root_path + f"\\{folder}"
    files = []
    for (dirpath, dirnames, filenames) in walk(path_f):
        for file in filenames:
            if file.find("frame")>-1:
                file_path = path_f + f"\\{file}"
                shutil.move(file_path, root_path)

