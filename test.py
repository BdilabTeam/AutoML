import os


dir = "./jobs"
file_list = os.listdir(dir)

for file_path in file_list:
    if file_path.find("-")!=-1:
        os.remove(dir+"/"+file_path)


