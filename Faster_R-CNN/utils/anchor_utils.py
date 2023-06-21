import torch
from torch import nn

def generate_anchors(scales, aspect_ratios, dtype=torch.float32, device="cpu"):
    scales = torch.as_tensor(scales, dtype=dtype, device=device)
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1.0 / h_ratios

    ws = (w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * scales[None, :]).view(-1)
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    return base_anchors.round()

class AnchorsGenerator(nn.Module):
    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def set_cell_anchors(self, dtype, device):
        """
        generate anchor template
        """
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is None

        cell_anchors = [generate_anchors(sizes, aspect_ratios, dtype, device) for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]
    
    def grid_anchors(self, feature_map_sizes, strides):
        """
        compute anchor coordinate list in origin image, mapped from feature map.
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        for size, stride, base_anchors in zip(feature_map_sizes, strides, cell_anchors):
            f_p_height, f_p_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            shifts_x = torch.arange(0, f_p_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, f_p_height, dtype=torch.float32, device=device) * stride_height

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors
    
    def cached_grid_anchors(self, feature_map_size, strides):
        """
        cached all anchor information
        """

        key = str(feature_map_size) + str(strides)
    
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(feature_map_size, strides)
        self._cache[key] = anchors
        
        return anchors

    def forward(self, image_list, feature_maps):
        """
        get feature map sizes
        """

        feature_map_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # get input image sizes
        image_size = image_list.tensors.shape[-2:]

        # get dtype and device
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # compute map stride between feature_maps and input images
        strides = [[torch.tensor(image_size[0] / g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] / g[1], dtype=torch.int64, device=device)] for g in feature_map_sizes]

        # get anchors template according size and aspect_ratios
        self.set_cell_anchors(dtype, device)

        # get anchor coordinate list in origin image, according to map
        anchors_over_all_feature_maps = self.cached_grid_anchors(feature_map_sizes, strides)

        anchors = []
        # for every image and feature map in a batch
        for i, (_, _) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # for every resolution feature map like fpn
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)

        # concat every resolution anchors, like fpn
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

        self._cache.clear()
        return anchors

            

if __name__ == "__main__":
    test = AnchorsGenerator()
    test.set_cell_anchors(dtype=torch.float32, device="cuda")