import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, SuperGlueForKeypointMatching


class SuperGlueMatcher:
    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.processor = AutoImageProcessor.from_pretrained(
            "magic-leap-community/superglue_outdoor"
        )
        self.model = (
            SuperGlueForKeypointMatching.from_pretrained(
                "magic-leap-community/superglue_outdoor"
            )
            .to(self.device)
            .eval()
        )

    def match(self, img_1, img_2, threshold: float = 0.5):
        imgs = [img_1, img_2]
        inputs = self.processor(imgs, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        img_sizes = [[(img.height, img.width) for img in imgs]]
        processed_outputs = self.processor.post_process_keypoint_matching(
            outputs, img_sizes, threshold=threshold
        )

        return processed_outputs

    def plot_samples(self, processed_outputs, imgs):
        return self.processor.visualize_keypoint_matching(imgs, processed_outputs)


def align_images_after_superglue(img_1, img_2, matched):
    src_pts = matched["keypoints1"].cpu().numpy()
    dst_pts = matched["keypoints0"].cpu().numpy()

    src_pts = src_pts.reshape(-1, 1, 2)
    dst_pts = dst_pts.reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=4.0
    )
    print(f"num inliers: {sum(mask)}")
    h, w = np.array(img_1).shape[:2]
    aligned = cv2.warpPerspective(np.array(img_2), H, (w, h))
    return aligned, H, mask
