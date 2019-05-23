import glob, os
path = "../data/COLLAR"
os.chdir(path)
for file in glob.glob("*.csv"):
    print(path + file)
