a = [1, 34, 53, 73, 2, 57, 34]


def q_sort(l, r):
    l0 = l
    r0 = r
    m = a[(l + r) // 2]
    while l<r:
        while a[l] < m and l<r:
            l += 1
        while a[r] > m and l<r:
            r -= 1
        a[l], a[r] = a[r], a[l]
        l += 1
        r -= 1
    if l0<r:
        q_sort(l0, r)
    if l<r0:
        q_sort(l, r0)

q_sort(0, len(a)-1)
print(a)