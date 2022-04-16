import torch

def distance2keypoints(center, distance, max_shape=None):
    assert distance.size(-1) == 8
    # 左上
    x1 = center[..., 0] - distance[..., 0]
    y1 = center[..., 1] - distance[..., 1]
    # 左下
    x2 = center[..., 0] - distance[..., 2]
    y2 = center[..., 1] + distance[..., 3]
    # 右下
    x3 = center[..., 0] + distance[..., 4]
    y3 = center[..., 1] + distance[..., 5]
    # 右上
    x4 = center[..., 0] + distance[..., 6]
    y4 = center[..., 1] - distance[..., 7]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
        x3 = x3.clamp(min=0, max=max_shape[1])
        y3 = y3.clamp(min=0, max=max_shape[0])
        x4 = x4.clamp(min=0, max=max_shape[1])
        y4 = y4.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], -1)

def keypoints2distance(center, keypoints, max_dis=None, eps=0.1):
    # 左上
    x1 = center[:, 0] - keypoints[:, 0]
    y1 = center[:, 1] - keypoints[:, 1]
    # 左下
    x2 = center[:, 0] - keypoints[:, 2]
    y2 = keypoints[:, 3] - center[:, 1]
    # 右下
    x3 = keypoints[:, 4] - center[:, 0]
    y3 = keypoints[:, 5] - center[:, 1]
    # 右上
    x4 = keypoints[:, 6] - center[:, 0]
    y4 = center[:, 1] - keypoints[:, 7]
    if max_dis is not None:
        x1 = x1.clamp(min=0, max=max_dis - eps)
        y1 = y1.clamp(min=0, max=max_dis - eps)
        x2 = x2.clamp(min=0, max=max_dis - eps)
        y2 = y2.clamp(min=0, max=max_dis - eps)
        x3 = x3.clamp(min=0, max=max_dis - eps)
        y3 = y3.clamp(min=0, max=max_dis - eps)
        x4 = x4.clamp(min=0, max=max_dis - eps)
        y4 = y4.clamp(min=0, max=max_dis - eps)
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], -1)
