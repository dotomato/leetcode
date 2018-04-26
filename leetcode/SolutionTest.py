import unittest
import Solution


class Test_SolutionTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.ins = Solution.Solution()
        return super().setUp(self)

    def test_461(self):
        result = self.ins.hammingDistance(1, 4)
        except_result = 2
        self.assertEqual(result, except_result)

    def test_657(self):
        result = self.ins.judgeCircle('UDRL')
        except_result = True
        self.assertEqual(result, except_result)

        result = self.ins.judgeCircle('UDRR')
        except_result = False
        self.assertEqual(result, except_result)

    def test_728(self):
        result = self.ins.selfDividingNumbers(1, 22)
        except_result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
        self.assertEqual(result, except_result)

    def test_561(self):
        result = self.ins.arrayPairSum([1, 4, 3, 2])
        except_result = 4
        self.assertEqual(result, except_result)

    def test_821(self):
        result = self.ins.shortestToChar('loveleetcode', 'e')
        except_result = [3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]
        self.assertEqual(result, except_result)

        result = self.ins.shortestToChar('aaba', 'b')
        except_result = [2, 1, 0, 1]
        self.assertEqual(result, except_result)

    def test_811(self):
        result = self.ins.subdomainVisits(["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"])
        except_result = ["901 mail.com", "50 yahoo.com", "900 google.mail.com", "5 wiki.org", "5 org",
                         "1 intel.mail.com", "951 com"]
        self.assertSetEqual(set(result), set(except_result))

    def test_806(self):
        result = self.ins.numberOfLines(
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            "abcdefghijklmnopqrstuvwxyz")
        except_result = [3, 60]
        self.assertEqual(result, except_result)

        result = self.ins.numberOfLines(
            [4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            "bbbcccdddaaa")
        except_result = [2, 4]
        self.assertEqual(result, except_result)

    def test_476(self):
        result = self.ins.findComplement(5)
        except_result = 2
        self.assertEqual(result, except_result)

    def test_786(self):
        result = self.ins.kthSmallestPrimeFraction([1, 2, 3, 5], 3)
        except_result = [2, 5]
        self.assertEqual(result, except_result)

        result = self.ins.kthSmallestPrimeFraction([1, 7], 1)
        except_result = [1, 7]
        self.assertEqual(result, except_result)

        result = self.ins.kthSmallestPrimeFraction([1, 13, 17, 59], 6)
        except_result = [13, 17]
        self.assertEqual(result, except_result)

    def test_500(self):
        result = self.ins.findWords(["Hello", "Alaska", "Dad", "Peace"])
        except_result = ["Alaska", "Dad"]
        self.assertEqual(result, except_result)

    def test_682(self):
        result = self.ins.calPoints(["5", "2", "C", "D", "+"])
        except_result = 30
        self.assertEqual(result, except_result)

        result = self.ins.calPoints(["5", "-2", "4", "C", "D", "9", "+", "+"])
        except_result = 27
        self.assertEqual(result, except_result)

    def test_575(self):
        result = self.ins.distributeCandies([1, 1, 2, 2, 3, 3])
        except_result = 3
        self.assertEqual(result, except_result)

        result = self.ins.distributeCandies([1, 1, 2, 3])
        except_result = 2
        self.assertEqual(result, except_result)

    def test_463(self):
        result = self.ins.islandPerimeter([[0, 1, 0, 0],
                                           [1, 1, 1, 0],
                                           [0, 1, 0, 0],
                                           [1, 1, 0, 0]])
        except_result = 16
        self.assertEqual(result, except_result)

    def test_766(self):
        result = self.ins.isToeplitzMatrix([[1, 2, 3, 4], [5, 1, 2, 3], [9, 5, 1, 2]])
        except_result = True
        self.assertEqual(result, except_result)

    def test_566(self):
        result = self.ins.matrixReshape([[1, 2], [3, 4]], 1, 4)
        except_result = [[1, 2, 3, 4]]
        self.assertEqual(result, except_result)

        result = self.ins.matrixReshape([[1, 2], [3, 4]], 2, 4)
        except_result = [[1, 2], [3, 4]]
        self.assertEqual(result, except_result)

    def test_496(self):
        result = self.ins.nextGreaterElement([4, 1, 2], [1, 3, 4, 2])
        except_result = [-1, 3, -1]
        self.assertEqual(result, except_result)

        result = self.ins.nextGreaterElement([2, 4], [1, 2, 3, 4])
        except_result = [3, -1]
        self.assertEqual(result, except_result)

    def test_693(self):
        result = self.ins.hasAlternatingBits(5)
        except_result = True
        self.assertEqual(result, except_result)

        result = self.ins.hasAlternatingBits(7)
        except_result = False
        self.assertEqual(result, except_result)

        result = self.ins.hasAlternatingBits(11)
        except_result = False
        self.assertEqual(result, except_result)

        result = self.ins.hasAlternatingBits(10)
        except_result = True
        self.assertEqual(result, except_result)

    def test_762(self):
        result = self.ins.countPrimeSetBits(6, 10)
        except_result = 4
        self.assertEqual(result, except_result)

        result = self.ins.countPrimeSetBits(10, 15)
        except_result = 5
        self.assertEqual(result, except_result)

        result = self.ins.countPrimeSetBits(842, 888)
        except_result = 23
        self.assertEqual(result, except_result)

    def test_796(self):
        result = self.ins.rotateString('abcde', 'cdeab')
        except_result = True
        self.assertEqual(result, except_result)

        result = self.ins.rotateString('abcde', 'abced')
        except_result = False
        self.assertEqual(result, except_result)

    def test_812(self):
        result = self.ins.largestTriangleArea([[0, 0], [0, 1], [1, 0], [0, 2], [2, 0]])
        except_result = 2
        self.assertEqual(result, except_result)

    def test_784(self):
        result = self.ins.letterCasePermutation("a1b2")
        except_result = ["a1b2", "A1b2", "a1B2", "A1B2"]
        self.assertEqual(result, except_result)

    def test_520(self):
        result = self.ins.detectCapitalUse("USA")
        except_result = True
        self.assertEqual(result, except_result)

    def test_695(self):
        result = self.ins.maxAreaOfIsland([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                           [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                                           [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])
        except_result = 6
        self.assertEqual(result, except_result)

    def test_389(self):
        nums = [0, 1, 0, 3, 12]
        self.ins.moveZeroes(nums)
        self.assertEqual(nums, [1, 3, 12, 0, 0])

    def test_283(self):
        result = self.ins.findTheDifference("abcd", "abdcd")
        except_result = "d"
        self.assertEqual(result, except_result)

    def test_791(self):
        result = self.ins.customSortString("cba", "abcd")
        except_result = "cbad"
        self.assertEqual(result, except_result)

    def test_696(self):
        result = self.ins.countBinarySubstrings("00110011")
        except_result = 6
        self.assertEqual(result, except_result)

        result = self.ins.countBinarySubstrings("10101")
        except_result = 4
        self.assertEqual(result, except_result)

    def test_819(self):
        result = self.ins.mostCommonWord("Bob hit a ball, the hit BALL flew far after it was hit.", ["hit"])
        except_result = "ball"
        self.assertEqual(result, except_result)

    def test_788(self):
        result = self.ins.rotatedDigits(10)
        except_result = 4
        self.assertEqual(result, except_result)

        result = self.ins.rotatedDigits(1)
        except_result = 0
        self.assertEqual(result, except_result)

        result = self.ins.rotatedDigits(857)
        except_result = 247
        self.assertEqual(result, except_result)

    def test_13(self):
        result = self.ins.romanToInt('III')
        except_result = 3
        self.assertEqual(result, except_result)

        result = self.ins.romanToInt('IV')
        except_result = 4
        self.assertEqual(result, except_result)

        result = self.ins.romanToInt('IX')
        except_result = 9
        self.assertEqual(result, except_result)

        result = self.ins.romanToInt('LVIII')
        except_result = 58
        self.assertEqual(result, except_result)

        result = self.ins.romanToInt('MCMXCIV')
        except_result = 1994
        self.assertEqual(result, except_result)

    def test_733(self):
        result = self.ins.floodFill([[1, 1, 1], [1, 1, 0], [1, 0, 1]], 1, 1, 2)
        except_result = [[2, 2, 2], [2, 2, 0], [2, 0, 1]]
        self.assertEqual(result, except_result)

    def test_122(self):
        result = self.ins.maxProfit([7, 1, 5, 3, 6, 4])
        except_result = 7
        self.assertEqual(result, except_result)

        result = self.ins.maxProfit([1, 2, 3, 4, 5])
        except_result = 4
        self.assertEqual(result, except_result)

        result = self.ins.maxProfit([7, 6, 4, 3, 1])
        except_result = 0
        self.assertEqual(result, except_result)

    def test_455(self):
        result = self.ins.findContentChildren([10, 9, 8, 7], [5, 6, 7, 8])
        except_result = 2
        self.assertEqual(result, except_result)

        result = self.ins.findContentChildren([1, 2, 3], [1, 1])
        except_result = 1
        self.assertEqual(result, except_result)

        result = self.ins.findContentChildren([1, 2], [1, 2, 3])
        except_result = 2
        self.assertEqual(result, except_result)

    def test_509(self):
        result = self.ins.findRelativeRanks([5, 4, 3, 2, 1])
        except_result = ["Gold Medal", "Silver Medal", "Bronze Medal", "4", "5"]
        self.assertEqual(result, except_result)

    def test_XXX(self):
        result = self.ins.XXXXXXXX()
        except_result = None
        self.assertEqual(result, except_result)


if __name__ == '__main__':
    unittest.main()
