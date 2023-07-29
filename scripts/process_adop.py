from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

def process_adop(scene_path: Path, checkpoint_path: Path, scales: List[int]) -> None:
    for scale in scales:
        (scene_path / f'undistorted_images_adop-{scale}').mkdir(exist_ok=True)

    ep_dirs = list(filter(lambda x: 'ep' in x.name, checkpoint_path.iterdir()))
    assert len(ep_dirs) == 1
    pose = list(filter(lambda x: 'poses' in x.name, (checkpoint_path / ep_dirs[0]).iterdir()))
    assert len(pose) == 1
    poses_path = pose[0]

    intrinsic = list(filter(lambda x: 'intrinsics' in x.name, (checkpoint_path / ep_dirs[0]).iterdir()))
    assert len(intrinsic) == 1
    intrinsics_path = intrinsic[0]

    point = list(filter(lambda x: 'points' in x.name, (checkpoint_path / ep_dirs[0]).iterdir()))
    assert len(point) == 1
    points_path = point[0]

    adop_points = torch.load(points_path, map_location='cpu')
    with (scene_path / 'adop-scene-bounds.txt').open('w') as f:
        min_bounds = adop_points.t_position[:, :3].min(dim=0)[0]
        max_bounds = adop_points.t_position[:, :3].max(dim=0)[0]
        f.write('{}\n{}\n'.format(' '.join([str(x.item()) for x in min_bounds]),
                                  ' '.join([str(x.item()) for x in max_bounds])))
        print(f'Scene bounds: {min_bounds} {max_bounds}')

    adop_poses = torch.load(poses_path, map_location='cpu').poses_se3
    min_near = 1e10
    max_far = -1
    with (scene_path / 'adop-poses.txt').open('w') as f:
        for pose in tqdm(adop_poses):
            position = pose[4:7]
            distance = (adop_points.t_position[:, :3] - position).norm(dim=-1)
            near = distance.min().item()
            far = distance.max().item()
            min_near = min(near, min_near)
            max_far = max(far, max_far)
            f.write('{} {} {}\n'.format(' '.join([str(x.item()) for x in pose[:7]]), near, far))
    print(f'Wrote {len(adop_poses)}. Near: {min_near}, far: {far}')

    adop_intrinsics = torch.load(intrinsics_path, map_location='cpu').intrinsics
    assert adop_intrinsics.shape == (1, 13)
    adop_intrinsics = adop_intrinsics.squeeze().detach()
    K = np.eye(3)
    K[0, 0] = adop_intrinsics[0]
    K[0, 1] = adop_intrinsics[4]
    K[1, 1] = adop_intrinsics[1]
    K[0, 2] = adop_intrinsics[2]
    K[1, 2] = adop_intrinsics[3]

    distortion = adop_intrinsics[5:].numpy()

    print('Writing undistorted images')
    new_intrinsics = []
    with (scene_path / 'images.txt').open() as f:
        for line in tqdm(f):
            image_path = scene_path / 'images' / line.strip()
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), 1, (w, h))
            dst = cv2.undistort(img, K, distortion, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            for scale in [1, 2, 4, 8]:
                if scale != 1:
                    cv2.imwrite(str(scene_path / f'undistorted_images_adop-{scale}' / image_path.name),
                                cv2.resize(dst, (dst.shape[1] // scale, dst.shape[0] // scale),
                                           interpolation=cv2.INTER_LANCZOS4))
                else:
                    cv2.imwrite(str(scene_path / f'undistorted_images_adop-{scale}' / image_path.name), dst)
            new_intrinsics.append([dst.shape[1], dst.shape[0]] + newcameramtx.reshape((-1)).tolist())

    with (scene_path / 'undistorted_intrinsics_adop.txt').open('w') as f:
        for i in new_intrinsics:
            f.write('{}\n'.format(' '.join([str(x) for x in i])))


def _get_opts() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--scene_path', type=Path, required=True)
    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--scales', type=list, default=[1, 2, 4, 8])

    return parser.parse_args()


def main(hparams: Namespace) -> None:
    process_adop(hparams.scene_path, hparams.checkpoint_path, hparams.scales)


if __name__ == '__main__':
    main(_get_opts())
