import numpy as np
from ..builder import PIPELINES

# TODO(barretzoph): Add in ArXiv link once paper is out.
def distort_image_with_autoaugment(image, bboxes, augmentation_name):
    """Applies the AutoAugment policy to `image` and `bboxes`.

    Args:
        image: `Tensor` of shape [height, width, 3] representing an image.
        bboxes: `Tensor` of shape [N, 4] representing ground truth boxes that are
            normalized between [0, 1].
        augmentation_name: The name of the AutoAugment policy to use. The available
            options are `v0`, `v1`, `v2`, `v3` and `test`. `v0` is the policy used for
            all of the results in the paper and was found to achieve the best results
            on the COCO dataset. `v1`, `v2` and `v3` are additional good policies
            found on the COCO dataset that have slight variation in what operations
            were used during the search procedure along with how many operations are
            applied in parallel to a single image (2 vs 3).

    Returns:
        A tuple containing the augmented versions of `image` and `bboxes`.
    """
    available_policies = {
        'v0': policy_v0,
        'v1': policy_v1,
        'v2': policy_v2,
        'v3': policy_v3,
        'v4': policy_v4,
        'test': policy_vtest
    }
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(
            augmentation_name))

    policy = available_policies[augmentation_name]()
    augmentation_hparams = {}
    return build_and_apply_nas_policy(policy, image, bboxes,
                                      augmentation_hparams)


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
