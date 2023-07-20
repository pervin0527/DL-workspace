import torch
from torch import nn

def generate_anchors(anchor_scales, aspect_ratios, dtype=torch.float32, device="cpu"):
    ## 중심점이 (0, 0)인 len(anchor_scales) * len(aspect_ratios)개의 anchor를 생성.
    anchor_scales = torch.as_tensor(anchor_scales, dtype=dtype, device=device)
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)

    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1.0 / h_ratios

    ## len(ratios) * len(scales)
    ws = (w_ratios[:, None] * anchor_scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * anchor_scales[None, :]).view(-1)
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    return base_anchors.round()


class AnchorsGenerator(nn.Module):
    def __init__(self, anchor_scales=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()
        self.anchor_scales = anchor_scales
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self.cache = {}

    def forward(self, image_list, feature_maps):
        feature_map_sizes = list((feature_map.shape[-2:] for feature_map in feature_maps))

        ## input image의 크기
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        ## input image와 feature map 간 stride를 계산.
        strides = [[torch.tensor(image_size[0] / g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] / g[1], dtype=torch.int64, device=device)] for g in feature_map_sizes]

        ## anchor scale과 aspect ratio에 따라 9개의 기본 anchor를 만듦.
        self.set_cell_anchors(dtype, device)

        ## feature map에 따라 원본 이미지에 적용될 anchor box들의 좌표들을 얻는다.
        anchors_over_all_feature_maps = self.cached_grid_anchors(feature_map_sizes, strides)

        anchors = []
        for i, (_, _) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)

        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

        self.cache.clear()
        return anchors
    
    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.anchor_scales, self.aspect_ratios)]

    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None

        cell_anchors = [generate_anchors(sizes, aspect_ratios, dtype, device) for sizes, aspect_ratios in zip(self.anchor_scales, self.aspect_ratios)]
        self.cell_anchors = cell_anchors

    def cached_grid_anchors(self, feature_map_size, strides):
        ## feature_map_size는 backbone으로부터 출력된 feature map의 크기
        ## 원본 이미지와 feature map간의 stride값.
        key = str(feature_map_size) + str(strides)
        
        if key in self.cache:
            return self.cache[key]
        
        anchors = self.grid_anchors(feature_map_size, strides)
        self.cache[key] = anchors
        
        return anchors
    
    def grid_anchors(self, feature_map_sizes, strides):
        ## feature map에 따라 원본 이미지에 적용될 anchor의 좌표들을 얻는다.
        ## feature_map_size는 backbone으로부터 출력된 feature map의 크기
        ## 원본 이미지와 feature map간의 stride값.

        anchors = []
        cell_anchors = self.cell_anchors  # anchor template
        assert cell_anchors is not None

        # for every resolution feature map, like fpn
        for size, stride, base_anchors in zip(feature_map_sizes, strides, cell_anchors):
            f_p_height, f_p_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            ## 원본 이미지에서 anchor의 x_center 좌표.
            shifts_x = torch.arange(0, f_p_width, dtype=torch.float32, device=device) * stride_width

            ## 원본 이미지에서 anchor의 y_center 좌표.
            shifts_y = torch.arange(0, f_p_height, dtype=torch.float32, device=device) * stride_height

            ## torch.meshgrid를 통해 원본 이미지에 N개의 anchor center 좌표를 만들어낸다.
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4)]



if __name__ == "__main__":
    generate_anchors((128, 256, 512), (0.5, 1.0, 2.0))