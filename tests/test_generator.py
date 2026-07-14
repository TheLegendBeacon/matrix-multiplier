import random, sys

n = int(sys.argv[1])
outpath = sys.argv[2]

with open(outpath, 'w') as f:
    f.write(f"{n} {n}\n")
    for _ in range(n):
        f.write(' '.join(f"{random.uniform(-1,1):.6f}" for _ in range(n)) + "\n")
    f.write("\n")
    f.write(f"{n} {n}\n")
    for _ in range(n):
        f.write(' '.join(f"{random.uniform(-1,1):.6f}" for _ in range(n)) + "\n")