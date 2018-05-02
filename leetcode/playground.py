import itertools
import collections
import heapq
import random

num = 195
x = num // 2
x2 = num // x
last_x = set()
while x not in last_x:
    last_x.add(x)
    x = (x + x2) // 2
    x2 = num // x
    print(x, x2)
print(num - x*x == 0)
