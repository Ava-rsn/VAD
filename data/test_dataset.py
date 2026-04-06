import glob
import os
import cv2
import numpy as np
import torch.utils.data
import json

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]


class AbnormalDatasetGradientsTest(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.ds_name = args.dataset
        self.input_3d = args.input_3d
        self.allowed_videos = None

        # split filtering should work for ANY dataset
        self.split = getattr(args, "split", None)          # "val" or "test"
        self.split_json = getattr(args, "split_json", None)

        if self.split_json is not None and self.split is not None:
            with open(self.split_json, "r") as f:
                split_data = json.load(f)

            if self.split == "val":
                self.allowed_videos = set(split_data["val_videos"])
            elif self.split == "test":
                self.allowed_videos = set(split_data["test_videos"])
            else:
                raise ValueError("split must be 'val' or 'test'")

        if args.dataset == "avenue":
            data_path = args.avenue_path
            gt_path = args.avenue_gt_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
            gt_path = args.shanghai_gt_path
        elif args.dataset == "ucf_crime":
            data_path = args.ucf_crime_path
            gt_path = args.ucf_crime_gt_path
        else:
            raise Exception("Unknown dataset!")

        # small cache so we don't reload the same npy file for every frame
        self._frames_cache = {}
        self._gradients_cache = {}

        self.data, self.labels, self.gradients = self._read_data(data_path, gt_path)

    
        loaded_videos = set()
        for x in self.data:
            if isinstance(x, tuple):
                loaded_videos.add(x[0])
            else:
                loaded_videos.add(x.split("/")[-2])
        print("Loaded videos:", len(loaded_videos))


    def _is_allowed_video(self, video_name):
        if self.allowed_videos is None:
            return True
        return self._normalize_video_name(video_name) in {
            self._normalize_video_name(v) for v in self.allowed_videos
        }
    

    def _load_gt(self, gt_path_base):
        npy_path = gt_path_base + ".npy"
        txt_path = gt_path_base + ".txt"

        if os.path.isfile(npy_path):
            return np.load(npy_path)
        elif os.path.isfile(txt_path):
            return np.loadtxt(txt_path)
        else:
            raise FileNotFoundError(f"Missing GT file: {npy_path} or {txt_path}")

    def _read_data(self, data_path, gt_path):
        data = []
        labels = []
        gradients = []

        test_root = os.path.join(data_path, "test")
        frames_npy_root = os.path.join(test_root, "frames")
        grads_npy_root = os.path.join(test_root, "gradients2")

        # New format: per-video npy files
        if os.path.isdir(frames_npy_root):
            video_files = sorted(glob.glob(os.path.join(frames_npy_root, "*.npy")))

            for video_file in video_files:
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                if self.allowed_videos is not None:
                    if video_name not in self.allowed_videos:
                        continue
                grad_file = os.path.join(grads_npy_root, f"{video_name}.npy")
                gt_base = os.path.join(gt_path, video_name)

                if not os.path.isfile(grad_file):
                    print(f"[WARN] Missing gradients for {video_name}, skipping")
                    continue

                try:
                    lbls = self._load_gt(gt_base)
                except FileNotFoundError:
                    print(f"[WARN] Missing GT for {video_name}, skipping")
                    continue

                frames = np.load(video_file, mmap_mode="r")
                grads_arr = np.load(grad_file, mmap_mode="r")

                n_frames = frames.shape[0]

                if grads_arr.shape[0] != n_frames:
                    print(
                        f"[WARN] Gradient/frame length mismatch for {video_name}: "
                        f"frames={n_frames}, grads={grads_arr.shape[0]}. Skipping"
                    )
                    continue

                if len(lbls) != n_frames:
                    print(
                        f"[WARN] GT/frame length mismatch for {video_name}: "
                        f"frames={n_frames}, gt={len(lbls)}. Skipping"
                    )
                    continue

                for i in range(n_frames):
                    # store logical frame references instead of file paths
                    data.append((video_name, i))
                    gradients.append((video_name, i))
                    labels.append(float(lbls[i]))

            return data, labels, gradients

        # Old format: folders of image files
        extension = None
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "test/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break

        if extension is None:
            raise RuntimeError(
                f"Could not find frames in either {frames_npy_root} or test/frames/*"
            )

        self.extension = extension
        dirs = sorted(glob.glob(os.path.join(data_path, "test", "frames", "*")))

        for dir_path in dirs:
            imgs_path = list(glob.glob(os.path.join(dir_path, f"*{extension}")))
            imgs_path = sorted(imgs_path, key=lambda x: int(os.path.basename(x).split('.')[0]))

            video_name = os.path.basename(dir_path)
            if not self._is_allowed_video(video_name):
                continue
            lbls = self._load_gt(os.path.join(gt_path, video_name))

            data += imgs_path
            labels += list(lbls)

            gradients_path = list(
                glob.glob(os.path.join(data_path, "test", "gradients2", video_name, "*.png"))
            )
            gradients_path = sorted(
                gradients_path, key=lambda x: int(os.path.basename(x).split('.')[0])
            )
            gradients += gradients_path

        return data, labels, gradients

    def _get_cached_npy(self, cache, path):
        if path not in cache:
            cache[path] = np.load(path, mmap_mode="r")
        return cache[path]

    def _read_frame_from_npy(self, video_name, frame_idx):
        frames_path = os.path.join(
            self._get_data_root(), "test", "frames", f"{video_name}.npy"
        )
        arr = self._get_cached_npy(self._frames_cache, frames_path)
        frame = np.asarray(arr[frame_idx])
        # stored as RGB, convert to BGR to match old cv2.imread behavior
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _read_gradient_from_npy(self, video_name, frame_idx):
        grads_path = os.path.join(
            self._get_data_root(), "test", "gradients2", f"{video_name}.npy"
        )
        arr = self._get_cached_npy(self._gradients_cache, grads_path)
        grad = np.asarray(arr[frame_idx])
        # stored as RGB, convert to BGR to match old cv2.imread behavior
        return cv2.cvtColor(grad, cv2.COLOR_RGB2BGR)

    def _get_video_length_from_npy(self, video_name):
        frames_path = os.path.join(
            self._get_data_root(), "test", "frames", f"{video_name}.npy"
        )
        arr = self._get_cached_npy(self._frames_cache, frames_path)
        return arr.shape[0]

    def _get_data_root(self):
        if self.ds_name == "avenue":
            return self.args.avenue_path
        elif self.ds_name == "shanghai":
            return self.args.shanghai_path
        elif self.ds_name == "ucf_crime":
            return self.args.ucf_crime_path
        else:
            raise Exception("Unknown dataset!")

    def __getitem__(self, index):
        sample_ref = self.data[index]

        # New npy-based format
        if isinstance(sample_ref, tuple):
            video_name, frame_idx = sample_ref
            n_frames = self._get_video_length_from_npy(video_name)

            current_img = self._read_frame_from_npy(video_name, frame_idx)

            prev_idx = max(0, frame_idx - 3)
            next_idx = min(n_frames - 1, frame_idx + 3)

            previous_img = self._read_frame_from_npy(video_name, prev_idx)
            next_img = self._read_frame_from_npy(video_name, next_idx)

            img = current_img
            if self.input_3d:
                img = np.concatenate([previous_img, current_img, next_img], axis=-1)

            gradient = self._read_gradient_from_npy(video_name, frame_idx)

            frame_id = f"{frame_idx:06d}"
            full_id = f"{video_name}/{frame_id}"

        # Old image-based format
        else:
            current_img = cv2.imread(self.data[index])
            dir_path, frame_no, len_frame_no = self.extract_meta_info(self.data, index)
            previous_img = self.read_prev_next_frame_if_exists(
                dir_path, frame_no, direction=-3, length=len_frame_no
            )
            next_img = self.read_prev_next_frame_if_exists(
                dir_path, frame_no, direction=3, length=len_frame_no
            )

            img = current_img
            if self.input_3d:
                img = np.concatenate([previous_img, current_img, next_img], axis=-1)

            gradient = cv2.imread(self.gradients[index])

            full_id = self.data[index]
            video_name = self.data[index].split("/")[-2]

        if img.shape[:2] != self.args.input_size[::-1]:
            img = cv2.resize(img, self.args.input_size[::-1])
            current_img = cv2.resize(current_img, self.args.input_size[::-1])
            gradient = cv2.resize(gradient, self.args.input_size[::-1])

        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        target = np.concatenate((current_img, mask), axis=-1)

        img = img.astype(np.float32)
        gradient = gradient.astype(np.float32)
        target = target.astype(np.float32)

        img = (img - 127.5) / 127.5
        target = (target - 127.5) / 127.5

        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
        gradient = np.swapaxes(gradient, 0, 1).swapaxes(0, -1)

        return img, gradient, target, self.labels[index], video_name, full_id

    def extract_meta_info(self, data, index):
        frame_no = int(data[index].split("/")[-1].split('.')[0])
        dir_path = "/".join(data[index].split("/")[:-1])
        len_frame_no = len(data[index].split("/")[-1].split('.')[0])
        return dir_path, frame_no, len_frame_no

    def read_prev_next_frame_if_exists(self, dir_path, frame_no, direction=-3, length=1):
        frame_path = dir_path + "/" + str(frame_no + direction).zfill(length) + self.extension
        if os.path.exists(frame_path):
            return cv2.imread(frame_path)
        else:
            return cv2.imread(dir_path + "/" + str(frame_no).zfill(length) + self.extension)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__