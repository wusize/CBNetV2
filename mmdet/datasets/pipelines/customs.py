import numpy as np
from ..builder import PIPELINES

from mmdet.datasets.pipelines.autoaug_utils import distort_image_with_autoaugment


@PIPELINES.register_module()
class CustomAutoAugment(object):
    def __init__(self, autoaug_type="v1"):
        """
        Args:
            autoaug_type (str): autoaug type, support v0, v1, v2, v3, test
        """
        super(CustomAutoAugment, self).__init__()
        self.autoaug_type = autoaug_type

    def __call__(self, results):
        """
        Learning Data Augmentation Strategies for Object Detection, see https://arxiv.org/abs/1906.11172
        """
        gt_bbox = results['gt_bboxes']
        im = results['img']
        if len(gt_bbox) == 0:
            return results

        # gt_boxes : [x1, y1, x2, y2]
        # norm_gt_boxes: [y1, x1, y2, x2]
        height, width, _ = im.shape
        norm_gt_bbox = np.ones_like(gt_bbox, dtype=np.float32)
        norm_gt_bbox[:, 0] = gt_bbox[:, 1] / float(height)
        norm_gt_bbox[:, 1] = gt_bbox[:, 0] / float(width)
        norm_gt_bbox[:, 2] = gt_bbox[:, 3] / float(height)
        norm_gt_bbox[:, 3] = gt_bbox[:, 2] / float(width)

        im, norm_gt_bbox = distort_image_with_autoaugment(im, norm_gt_bbox,
                                                          self.autoaug_type)
        gt_bbox[:, 0] = norm_gt_bbox[:, 1] * float(width)
        gt_bbox[:, 1] = norm_gt_bbox[:, 0] * float(height)
        gt_bbox[:, 2] = norm_gt_bbox[:, 3] * float(width)
        gt_bbox[:, 3] = norm_gt_bbox[:, 2] * float(height)

        results['gt_bboxes'] = gt_bbox
        results['img'] = im
        results['img_shape'] = im.shape
        results['pad_shape'] = im.shape

        return results
