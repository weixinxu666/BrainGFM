import numpy as np


def random_mask_fmri(time_series, mask_ratio=0.2, mask_length=10, num_masks=3):
    """
    对 fMRI time series 进行随机 mask。

    :param time_series: np.ndarray, shape=(roi_num, time_points), fMRI 时间序列数据
    :param mask_ratio: float, 被 mask 的 ROI 数量比例
    :param mask_length: int, 每个 ROI 被 mask 的片段长度
    :param num_masks: int, 每个 ROI 需要 mask 的片段数量
    :return: np.ndarray, mask 处理后的 time series
    """
    roi_num, time_points = time_series.shape
    masked_data = time_series.copy()

    # 计算需要 mask 的 ROI 数量
    mask_roi_num = int(roi_num * mask_ratio)
    mask_rois = np.random.choice(roi_num, mask_roi_num, replace=False)

    for roi in mask_rois:
        for _ in range(num_masks):
            # 确保 mask 片段不会超出时间序列长度
            if time_points <= mask_length:
                start_idx = 0
            else:
                start_idx = np.random.randint(0, time_points - mask_length + 1)

            masked_data[roi, start_idx:start_idx + mask_length] = 0

    return masked_data


if __name__ == '__main__':


    # 示例数据
    roi_num = 100  # 100 个 ROI
    time_points = 200  # 每个 ROI 200 个时间点
    time_series = np.random.rand(roi_num, time_points)  # 生成随机 fMRI 时间序列

    # 应用随机 mask
    masked_series = random_mask_fmri(time_series, mask_ratio=0.2, mask_length=10, num_masks=3)
    print(masked_series)
