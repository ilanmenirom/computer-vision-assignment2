from solution import Solution
import numpy as np

a = np.arange(40).reshape((4,5,2))
num_rows, num_cols, _ = a.shape
d = Solution._manipulate_ssdd_diagonally(a)
res = Solution.inverse_diagonal(d[:, :, 0], num_rows, num_cols)
print(res)