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
       except_result = ["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
       self.assertSetEqual(set(result), set(except_result))

    def test_806(self):
       result = self.ins.numberOfLines([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], "abcdefghijklmnopqrstuvwxyz")
       except_result = [3, 60]

       result = self.ins.numberOfLines([4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], "bbbcccdddaaa")
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
       result = self.ins.calPoints(["5","2","C","D","+"])
       except_result = 30
       self.assertEqual(result, except_result)

       result = self.ins.calPoints(["5","-2","4","C","D","9","+","+"])
       except_result = 27
       self.assertEqual(result, except_result)

    def test_575(self):
       result = self.ins.distributeCandies([1,1,2,2,3,3])
       except_result = 3
       self.assertEqual(result, except_result)

       result = self.ins.distributeCandies([1,1,2,3])
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
       result = self.ins.isToeplitzMatrix([[1,2,3,4],[5,1,2,3],[9,5,1,2]])
       except_result = True
       self.assertEqual(result, except_result)


    def test_566(self):
        result = self.ins.matrixReshape([[1,2], [3,4]], 1, 4)
        except_result = [[1,2,3,4]]
        self.assertEqual(result, except_result)

        result = self.ins.matrixReshape([[1,2], [3,4]], 2, 4)
        except_result = [[1,2], [3,4]]
        self.assertEqual(result, except_result)


    def test_496(self):
        result = self.ins.nextGreaterElement([4,1,2], [1,3,4,2])
        except_result = [-1,3,-1]
        self.assertEqual(result, except_result)

        result = self.ins.nextGreaterElement([2,4], [1,2,3,4])
        except_result = [3,-1]
        self.assertEqual(result, except_result)


if __name__ == '__main__':
    unittest.main()
