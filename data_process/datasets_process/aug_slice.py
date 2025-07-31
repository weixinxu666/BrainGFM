import numpy as np

def slice_matrix(matrix):
    num_points, t = matrix.shape
    slice_length = 100

    if t <= slice_length:
        return [matrix]

    # 根据长度t确定切片数量
    num_slices = (t + slice_length - 1) // slice_length
    overlap = (num_slices * slice_length - t) // (num_slices - 1)

    slices = []
    for i in range(num_slices):
        start = i * (slice_length - overlap)
        end = start + slice_length
        if end > t:  # 确保所有切片长度为100
            start = t - slice_length
            end = t
        slices.append(matrix[:, start:end])
        print(start, end)

    return slices


if __name__ == '__main__':


    # 示例
    matrix = np.random.rand(100, 380)  # 生成一个100 x 350的矩阵
    sliced_matrices = slice_matrix(matrix)
    for idx, sm in enumerate(sliced_matrices):
        print(f"Slice {idx+1} shape: {sm.shape}")
