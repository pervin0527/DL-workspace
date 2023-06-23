import numpy as np

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """
    주어진 aspect_ratio와 scale값들을 열거하여 base window를 만든다.
    각각의 scale에 대한 aspect_ratio를 반영.(하나의 scale 당 3개의 window가 생성되므로 총 9개.)
    """
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in np.arange(len(ratios)):
        for j in np.arange(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.

    return anchor_base

if __name__ == "__main__":
    test = generate_anchor_base()
    print(test)