import csv


def str2list(s):
    s = s.replace(" ", "")
    s1 = s.split(",")
    print(s1)
    sc0 = s1[0]
    sc1 = s1[1]
    sc2 = s1[2]
    color0 = int(sc0[1:])
    print(color0)
    color1 = int(sc1)
    print(color1)
    color2 = int(sc2[: -1])
    return [color0, color1, color2]

path = "D:\SemiTracker\semitracker\datasets\annotation_test\csv\csv_0.csv"

dic = {1: [12, 33, 45], 2: [12, 33, 45], 3: [12, 33, 45], 5: [12, 33, 45]}

with open("csv_0.csv",'w',newline="")as f:
    fieldnames = [key for key in dic.keys()]
    writer=csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(dic)

with open("csv_0.csv",'r')as f:
    fieldnames = [key for key in dic.keys()]
    reader = csv.DictReader(f, fieldnames=fieldnames)
    for i, line in enumerate(reader):
        if i == 1:
            print(line)
            s = line[1]
            print(s)
            print(str2list(s))






