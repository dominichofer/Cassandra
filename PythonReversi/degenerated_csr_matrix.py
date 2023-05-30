class CSRMatrix:
    def __init__(self, values, col_indices, row_indices):
        self.values = values
        self.col_indices = col_indices
        self.row_indices = row_indices
        
    def matvec(self, vector):
        result = [0] * (len(self.row_indices) - 1)
        for i in range(len(self.row_indices) - 1):
            for j in range(self.row_indices[i], self.row_indices[i+1]):
                result[i] += self.values[j] * vector[self.col_indices[j]]
        return result


class CSRMatrix2:
    def __init__(self, col_indices, elements_per_row):
        self.col_indices = col_indices
        self.elements_per_row = elements_per_row
        
    def matvec(self, vector):
        rows = int(len(self.col_indices) / self.elements_per_row)
        result = [0] * rows
        for i in range(rows):
            for j in range(i * self.elements_per_row, (i + 1) * self.elements_per_row):
                result[i] += vector[self.col_indices[j]]
        return result


# Matrix
# 3 0 0
# 0 0 3
# 0 1 2
mat1 = CSRMatrix([3,3,1,2],[0,2,1,2],[0,1,2,4])
mat2 = CSRMatrix2([0,0,0,2,2,2,1,2,2],3)

vec = [1,2,3]

print(mat1.matvec(vec))
print(mat2.matvec(vec))