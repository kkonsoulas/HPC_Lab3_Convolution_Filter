import sys

def csvAppendToRow(csv,row,element):
    csv = open(csv,"r+")
    lines = csv.readlines()
    line = lines[row]
    line.replace('\n','')
    line = line + element + ','
    # line.strip()
    # print(lines)
    lines[row] = line
    # print("After:")
    # print(lines)
    csv.seek(0)
    csv.write("".join(lines))
    return

with open('out.txt') as f:
    lines = f.readlines()

cpuline  = lines[-2]
cpuline = cpuline.split()
cputime = cpuline[-2]

gpuline = lines[-1]
gpuline = gpuline.split()
gputime = gpuline[-2]


csvAppendToRow("cpu.csv",int(sys.argv[1]),cputime)
csvAppendToRow("gpu.csv",int(sys.argv[1]),gputime)
