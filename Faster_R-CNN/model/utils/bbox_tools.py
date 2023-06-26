import numpy as np

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """
    주어진 aspect_ratio와 scale값들을 열거하여 base window를 만든다.
    각각의 scale에 대한 aspect_ratio를 반영.(하나의 scale 당 3개의 window가 생성되므로 총 9개.)
    """
    py = base_size / 2.
    px = base_size / 2.

    ## scale 3개, ratio 3개 총 9개의 anchor를 만들어내며, [cx, cy, w, h] 4개의 값으로 구성.
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in np.arange(len(ratios)):
        for j in np.arange(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i]) ## 8 * scale * sqrt(ratio)
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i]) ## 8 * scale * sqrt(1 / ratio)

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.

    return anchor_base


def loc2bbox(src_bbox, loc):
    """
    bounding box offset과 scale 값들을 이용해서 bbox들을 decoding한다.
    bbox2loc에 의해 계산된 bbox offset과 scale이 주어지면 이 함수는 표현을 2D 이미지 좌표의 좌표로 decoding한다.
    주어진 scale과 offset들은 t_y, t_x, t_h, t_w(target)과 p_y, p_x, p_h, p_w(prediction)
    """
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """
    source bbox와 target bbox들 간에 주어진 bbox들에서 이 함수를 통해 offset과 scale을 계산한다.
    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()

    return loc


def bbox_iou(bbox_a, bbox_b):
    """
    bounding box들 간에 IoU를 계산한다.
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    ## top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    ## bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    
    return area_i / (area_a[:, None] + area_b - area_i)

if __name__ == "__main__":
    test = generate_anchor_base()
    print(test.shape)
    print(test)