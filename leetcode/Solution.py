import math, itertools
class TreeNode:
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

class Employee:
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates

class Solution(object):
    """leetcode"""

    def XXXXXXXX(self):
        return None

    def hammingDistance(self, x, y):
        
        def numtobi(n):
            n = int(n)
            s = []
            while n != 0:
                s.append(n % 2)
                n //= 2
            return s[::-1]

        xs = numtobi(x)
        ys = numtobi(y)

        if len(xs) < len(ys):
            xs, ys = ys, xs

        l_xs = len(xs)
        l_ys = len(ys)
        for i in range(l_xs - l_ys):
            ys.insert(0, 0)

        dis = 0
        for i in range(l_xs):
            if xs[i] != ys[i]:
                dis += 1


        return dis

    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        position = [0, 0]

        for i in moves:
            if i == 'U':
                position[1] += 1
            elif i == 'D':
                position[1] -= 1
            elif i == 'L':
                position[0] -= 1
            elif i == 'R':
                position[0] += 1

        return position[0] == 0 and position[1] == 0
    
    def mergeTrees(self, t1, t2):
        if not t1 and not t2:
            return None
        ans = TreeNode(t1.val if t1 else 0) + (t2.val if t2 else 0)
        ans.left = self.mergeTrees(t1 and t1.left, t2 and t2.left)
        ans.right = self.mergeTrees(t1 and t1.right, t2 and t2.right)

    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """

        result = []
        for i in range(left, right+1):

            digit = []
            num = i
            while num != 0:
                digit.append(num % 10)
                num = num // 10

            flag = True
            for j in digit:
                if j == 0 or i % j != 0:
                    flag = False
                    break

            if flag:
                result.append(i)

        return result

    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        nums.sort()
        s = sum(nums[::2])
        return s

    def shortestToChar(self, S, C):
        """
        :type S: str
        :type C: str
        :rtype: List[int]
        """
        b = []
        for i, s in enumerate(S):
            if s == C:
                b.append(i)


        a = list(range(b[0], -1, -1))

        for i in range(len(b)-1):
            l = b[i+1] - b[i]
            a.extend(range(1, l//2+1))
            a.extend(range((l-1)//2, -1, -1))

        a.extend(range(1, len(S) - b[-1]))

        return a

    def subdomainVisits(self, cpdomains):
        """
        :type cpdomains: List[str]
        :rtype: List[str]
        """

        domain_count = {}
        for domain_pair in cpdomains:
            count, domain = domain_pair.split(' ')
            count = int(count)
            domain_array = domain.split('.')
            for i in range(len(domain_array)):
                sub_domain = '.'.join(domain_array[i:])
                if sub_domain in domain_count:
                    domain_count[sub_domain] = domain_count[sub_domain] + count  
                else:
                    domain_count[sub_domain] = count

        result = []
        for key in domain_count.keys():
            result.append('{} {}'.format(domain_count[key], key))

        return result

    def numberOfLines(self, widths, S):
        """
        :type widths: List[int]
        :type S: str
        :rtype: List[int]
        """

        s = [ord(x) - ord('a') for x in S]
        y = 1
        x = 0
        for i in s:
            if x + widths[i] > 100:
                y += 1
                x = widths[i]
            else:
                x += widths[i]

        return [y, x]

    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """

        s = []
        while num != 0:
            s.append(num % 2)
            num = num // 2
        result = 0
        for i in s[::-1]:
            result = result << 1
            if i == 0:
                result += 1

        return result

    def kthSmallestPrimeFraction(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: List[int]
        """

        A.sort()
        n = len(A)
        
        l = 0
        r = 1

        while l < r:
            m = (l + r) / 2
            cnt = 0
            j = n-1
            max_pq = -1
            max_p = 0
            max_q = 0
            for i in range(n-2, -1, -1):
                pq =  A[i] / A[j]
                while j >= i and pq < m:
                    if pq > max_pq:
                        max_pq = pq
                        max_p = i
                        max_q = j
                    j -= 1
                    pq =  A[i] / A[j]
                cnt += n - 1 - j
            
            if cnt < K:
                l = m
            elif cnt > K:
                r = m
            else:
                return [A[max_p], A[max_q]]

        if l == r:
            return [A[0], A[1]]
        
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        
        keyboard = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        keyboard_dict = {}
        for i, item in enumerate(keyboard):
            for c in item:
                keyboard_dict[c] = i
            for c in item.upper():
                keyboard_dict[c] = i

        result = []
        for word in words:
            for i in range(len(word)-1):
                if keyboard_dict[word[i]] != keyboard_dict[word[i+1]]:
                    break
            else:
                result.append(word)
        return result
        
    def trimBST(self, root, L, R):

        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        if root is None:
            return None
        root.left = (root.left)
        root.right = trimBST(root.right)
        if L > root.val:
            return trimBSTroot.right
        elif R < root.val:
            return root.left
        else:
            return root

    def calPoints(self, ops):
        """
        :type ops: List[str]
        :rtype: int
        """
        points = []

        for op in ops:
            if op == 'C':
                points.pop()
            elif op == 'D':
                points.append(points[-1] * 2)
            elif op == '+':
                points.append(points[-1] + points[-2])
            else:
                points.append(int(op))

        return sum(points)

    def distributeCandies(self, candies):
        """
        :type candies: List[int]
        :rtype: int
        """
        n_h2 = len(candies) // 2
        candy_kind = len(set(candies))
        if candy_kind<n_h2:
            return candy_kind
        else:
            return n_h2

    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []
        for k in range(n):
            i = k+1
            if i % 15 == 0 :
                result.append('FizzBuzz')
            elif i % 5 == 0:
                result.append('Buzz')
            elif i % 3 == 0:
                result.append('Fizz')
            else:
                result.append(str(i))

        return result

    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        height = len(grid)
        width = len(grid[0])

        perimeter = 0
        for i in range(height):
            for j in range(width):
                if grid[i][j] == 1:
                    if i == 0:
                       perimeter += 1
                    else:
                        if grid[i-1][j] == 0:
                            perimeter += 1

                    if i == height-1:
                        perimeter += 1
                    else:
                        if grid[i+1][j] == 0:
                            perimeter += 1

                    if j == 0:
                        perimeter += 1
                    else:
                        if grid[i][j-1] == 0:
                            perimeter += 1

                    if j == width-1:
                        perimeter += 1
                    else:
                        if grid[i][j+1] == 0:
                            perimeter += 1

        return perimeter

    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        m = len(matrix)
        n = len(matrix[0])
        mn = max(m, n)
        for i in range(0, mn):
            for j in range(1, mn-i):
                if j < m and i+j < n and matrix[j][i+j] != matrix[j-1][i+j-1]:
                    return False
        for i in range(1, mn):
            for j in range(1, mn-i):
                if i+j < m and j < n and matrix[i+j][j] != matrix[i+j-1][j-1]:
                    return False
        return True

    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        r0 = len(nums)
        c0 = len(nums[0])
        if r0 * c0 != r * c:
            return nums

        result = []
        x = 0
        y = 0
        for i in range(r):
            line = []
            for j in range(c):
                line.append(nums[x][y])
                if y == c0 - 1:
                    x += 1
                    y = 0
                else:
                    y += 1
            result.append(line)
        return result

    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        x_dict = {}
        stack = []
        for n in nums2:
            while len(stack) != 0 and stack[-1] < n:
                x_dict[stack.pop()] = n
            stack.append(n)
            x_dict[n] = -1
        nums1 = [x_dict[n] for n in nums1]
        return nums1

    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """

        result = []
        a = [root]
        while len(a) != 0:
            vals = [i.val for i in a]
            result.append(sum(vals) / len(vals))
            b = []
            for i in a:
                if i.left is not None:
                    b.append(i.left)
                if i.right is not None:
                    b.append(i.right)
            a = b
        return result

    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        a = []
        while n != 0:
            a.append(n % 2)
            n = n // 2
        for i in range(len(a)-1):
            if a[i] == a[i+1]:
                return False
        return True

    def countPrimeSetBits(self, L, R):
        """
        :type L: int
        :type R: int
        :rtype: int
        """

        def prime(n):
            if n == 1 or n == 0:
                return False
            n_2 = int(math.sqrt(n))
            for i in range(2, n_2 + 1):
                if n % i == 0:
                    return False
            else:
                return True

        def num2bi(n):
            a = []
            while n != 0:
                a.append(n % 2)
                n = n >> 1
            return a

        prime_ar = [prime(i) for i in range(0, 40)]

        bi = num2bi(L)
        count = sum(bi)
        result = 1 if prime_ar[count] else 0

        for i in range(L, R):
            k = 0
            len_bi = len(bi)
            while k < len_bi and bi[k] == 1:
                bi[k] = 0
                count -= 1
                k += 1
            if k == len_bi:
                bi.append(1)
            else:
                bi[k] = 1
            count += 1
            if prime_ar[count]:
                result += 1
        return result

    def rotateString(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: bool
        """

        len_B = len(B)
        A = A + A
        for i in range(len_B):
            k = 0
            for j in range(len_B):
                if A[i+j] == B[j]:
                    k += 1
                else:
                    break
            if k == len_B:
                return True
        else:
            return False

    def largestTriangleArea(self, points):
        """
        :type points: List[List[int]]
        :rtype: float
        """
        result = 0
        l = len(points)
        for i in range(l-2):
            for j in range(i+1, l-1):
                for k in range(j+1, l):
                    xa = points[i][0]
                    ya = points[i][1]
                    xb = points[j][0]
                    yb = points[j][1]
                    xc = points[k][0]
                    yc = points[k][1]

                    area = 0.5*abs(xa*yb + xb*yc + xc*ya - xa*yc - xc*yb - xb*ya)
                    result = max(result, area)
        return result

    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        '''
        L = [[i.lower(), i.upper()] if i.isalpha() else i for i in S]
        return [''.join(i) for i in itertools.product(*L)]
        '''
        S = list(S.lower())
        index = [i for i in range(len(S)) if not S[i].isdigit()]
        l = len(index)
        bi = [0]*l
        result = []
        for i in range(1 << (len(index))):
            result.append(''.join(S))
            k = 0
            while k < l and bi[k] == 1:
                bi[k] = 0
                S[index[k]] = S[index[k]].lower()
                k += 1
            if k != l:
                bi[k] = 1
                S[index[k]] = S[index[k]].upper()
        return result

    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        count = sum([1 if x.isupper() else 0 for x in word])
        if len(word) == count or count == 0:
            return True
        else:
            return word[0].isupper() and count == 1

    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        if num == 0:
            return 0
        else:
            result = num % 9
            if result == 0:
                return 9
            else:
                return result

    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        result= 0
        m = len(grid)
        n = len(grid[0])
        shift = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    grid[i][j] = 0
                    count = 1
                    q = [(i, j)]
                    while len(q) != 0:
                        cod = q.pop()
                        for cod_shit in shift:
                            new_cod = (cod[0]+cod_shit[0], cod[1]+cod_shit[1])
                            if 0<=new_cod[0]<m and 0<=new_cod[1]<n and grid[new_cod[0]][new_cod[1]]:
                                q.append(new_cod)
                                count += 1
                                grid[new_cod[0]][new_cod[1]] = 0
                    result = max(result, count)
        return result

    def getImportance(self, employees, id):
        """
        :type employees: list[Employee]
        :type id: int
        :rtype: int
        """
        employees_dict = {}
        for em in employees:
            employees_dict[em.id] = em

        def get1(id1):
            if len(employees_dict[id1].subordinates) == 0:
                return employees_dict[id1].importance
            else:
                return sum(get1(i) for i in employees_dict[id1].subordinates) + employees_dict[id1].importance

        return get1(id)

    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i = 0
        l = len(nums)
        while i < l and nums[i] != 0:
            i += 1
        j = i
        while j < l and nums[j] == 0:
            j += 1
        while j != l:
            nums[i] = nums[j]
            nums[j] = 0

            i += 1
            while j < l and nums[j] == 0:
                j += 1
        return nums

    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        s_dict = {}
        for c in s:
            if c in s_dict:
                s_dict[c] += 1
            else:
                s_dict[c] = 1

        t_dict = {}
        for c in t:
            if c in t_dict:
                t_dict[c] += 1
            else:
                t_dict[c] = 1

        for c in t_dict.keys():
            if c not in s_dict or s_dict[c] != t_dict[c]:
                return c

    def customSortString(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: str
        """
        index = {}
        for i, c in enumerate(S):
            index[c] = i
        t1 = [c for c in T if c in S]
        t2 = [c for c in T if c not in S]

        t1.sort(key=lambda x: index[x])
        return ''.join(t1+t2)
