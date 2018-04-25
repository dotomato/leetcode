import itertools

for num in range(10000):
    result1 = num % 9
    if result1 == 0:
        result1 = 9
    if num == 0:
        result1 = 0
    a = list(str(num))
    while len(a) != 1:
        b = [int(x) for x in a]
        a = list(str(sum(b)))
    result2 = int(a[0])
    if result1 != result2:
        print(result1, result2)