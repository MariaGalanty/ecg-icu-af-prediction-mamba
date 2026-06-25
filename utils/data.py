import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset

"""
def get_record_paths_labels_patients(af_dir, sr_dir):

    Scan the two directories and build:
      - file_paths: list of base paths (without extension)
      - labels: list of 0/1 labels (1 = AF, 0 = SR)
      - patient_ids: list of patient IDs (string), derived from filename prefix before '_'
                     e.g., '00001_20220101_2211' -> patient_id = '00001'
    Expect matching .dat and .hea files.

    file_paths = []
    labels = []
    patient_ids = []

    for dir_path, label in [(af_dir, 1), (sr_dir, 0)]:
        if not os.path.isdir(dir_path):
            print(f"Warning: directory not found: {dir_path}")
            continue

        for fname in os.listdir(dir_path):
            if fname.lower().endswith(".dat"):
                base = os.path.splitext(fname)[0]  # e.g., '00001_20220101_2211'
                patient_id = base.split("_")[0]  # '00001'

                base_path = os.path.join(dir_path, base)
                dat_path = base_path + ".dat"
                hea_path = base_path + ".hea"

                if not os.path.exists(hea_path):
                    print(f"Warning: .hea file missing for {dat_path}, skipping.")
                    continue

                file_paths.append(base_path)
                labels.append(label)
                patient_ids.append(patient_id)

    file_paths = np.array(file_paths)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)

    print(f"Total records found: {len(file_paths)}")
    print(f"  AF (1) records: {np.sum(labels == 1)}")
    print(f"  SR (0) records: {np.sum(labels == 0)}")
    print(f"  Unique patients: {len(np.unique(patient_ids))}")

    return file_paths, labels, patient_ids
"""


def get_record_paths_labels_patients(af_dir, sr_dir, dataset_prefix=""):
    """
    Scan the two directories and build:
      - file_paths: list of base paths (without extension)
      - labels: list of 0/1 labels (1 = AF, 0 = SR)
      - patient_ids: list of patient IDs (string), derived from filename prefix before '_'
                     e.g., '00001_20220101_2211' -> patient_id = '<prefix>00001'
    Expect matching .dat and .hea files.
    """
    file_paths = []
    labels = []
    patient_ids = []

    for dir_path, label in [(af_dir, 1), (sr_dir, 0)]:
        if not os.path.isdir(dir_path):
            print(f"Warning: directory not found: {dir_path}")
            continue

        for fname in os.listdir(dir_path):
            if fname.lower().endswith(".dat"):
                base = os.path.splitext(fname)[0]  # e.g., '00001_20220101_2211'
                raw_pid = base.split("_")[0]  # '00001'
                patient_id = f"{dataset_prefix}{raw_pid}"

                base_path = os.path.join(dir_path, base)
                dat_path = base_path + ".dat"
                hea_path = base_path + ".hea"

                if not os.path.exists(hea_path):
                    print(f"Warning: .hea file missing for {dat_path}, skipping.")
                    continue

                file_paths.append(base_path)
                labels.append(label)
                patient_ids.append(patient_id)

    file_paths = np.array(file_paths)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)

    print(f"Total records found in {dataset_prefix or 'NO_PREFIX'}:")
    print(f"  AF (1) records: {np.sum(labels == 1)}")
    print(f"  SR (0) records: {np.sum(labels == 0)}")
    print(f"  Unique patients: {len(np.unique(patient_ids))}")

    return file_paths, labels, patient_ids


class SignalWindowDatasetEqual(Dataset):
    def __init__(
        self,
        file_paths,
        labels,
        window_minutes=10,
        fs_override=None,
        overlap_minutes=0,
        downsample_factor=1,
        max_windows_per_record=3,
        random_state=42,
        augment=False,
    ):
        """
        Args:
            # TO FILL
            max_windows_per_record: maximum number of windows to sample per record
            augment: whether to apply data augmentation (for train only)
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        if isinstance(labels, (int, float)):
            labels = [labels]

        assert len(file_paths) == len(labels), (
            "file_paths and labels must have same length"
        )

        self.file_paths = list(file_paths)
        self.file_labels = list(labels)
        self.max_windows_per_record = max_windows_per_record
        self.random_state = random_state
        self.augment = augment

        random.seed(self.random_state)

        # Lazy: store (dat_path, start_sample, window_size) — no signal in RAM
        self._items = []       # (dat_path, start, window_size)
        self.labels = []
        self.record_index = []

        for rec_idx, (path, file_label) in enumerate(
            zip(self.file_paths, self.file_labels)
        ):
            label = int(file_label)

            fs = fs_override
            hea_path = path + ".hea"
            if os.path.exists(hea_path):
                with open(hea_path, "r") as f:
                    lines = f.readlines()
                    if fs is None and len(lines) > 0:
                        first_line = lines[0].split()
                        if len(first_line) >= 3:
                            try:
                                fs = float(first_line[2])
                            except ValueError:
                                fs = 500.0
            if fs is None:
                fs = 500.0

            dat_path = path + ".dat"
            if not os.path.exists(dat_path):
                print(f"Error: .dat file not found: {dat_path}")
                continue

            if downsample_factor > 1:
                effective_fs = fs / downsample_factor
            else:
                effective_fs = fs

            window_size = int(window_minutes * 60 * effective_fs)
            stride = int((window_minutes - overlap_minutes) * 60 * effective_fs)

            # File length in samples after downsampling (float32 = 4 bytes each)
            n_samples = os.path.getsize(dat_path) // 4 // downsample_factor

            if n_samples < window_size:
                print(f"Warning: signal in {path} shorter than window size. Skipping.")
                continue

            possible_starts = list(range(0, n_samples - window_size + 1, stride))

            if len(possible_starts) == 0:
                continue

            num_to_select = min(self.max_windows_per_record, len(possible_starts))
            selected_starts = random.sample(possible_starts, num_to_select)

            for start in selected_starts:
                self._items.append((dat_path, start, window_size, downsample_factor))
                self.labels.append(label)
                self.record_index.append(rec_idx)

        self.labels = np.array(self.labels, dtype=np.float32)
        self.record_index = np.array(self.record_index, dtype=np.int32)

        # Try to load all windows into RAM for fast __getitem__.
        # Fall back to lazy memmap if there isn't enough memory.
        self._windows = None
        try:
            import psutil
            # Estimate memory needed: n_windows * window_size * 4 bytes
            n_windows = len(self._items)
            window_size = self._items[0][2] if n_windows > 0 else 0
            estimated_bytes = n_windows * window_size * 4
            available_bytes = psutil.virtual_memory().available
            if estimated_bytes > available_bytes * 0.4:
                raise MemoryError(
                    f"Skipping eager load: estimated {estimated_bytes/1e9:.1f} GB "
                    f"> 40% of available {available_bytes/1e9:.1f} GB"
                )
            # Pre-allocate to avoid the 2x memory spike from list→np.array conversion
            self._windows = np.empty((n_windows, window_size), dtype=np.float32)
            for i, (dat_path, start, win_size, ds_factor) in enumerate(self._items):
                mm = np.memmap(dat_path, dtype=np.float32, mode='r')
                if ds_factor > 1:
                    raw_start = start * ds_factor
                    chunk = mm[raw_start: raw_start + win_size * ds_factor: ds_factor]
                else:
                    chunk = mm[start: start + win_size]
                std = chunk.std()
                self._windows[i] = (chunk - chunk.mean()) / (std + 1e-8)
                del mm
            print(
                f"Created dataset: {len(self.labels)} windows "
                f"({np.sum(self.labels == 1)} AF, {np.sum(self.labels == 0)} SR) "
                f"from {len(self.file_paths)} records "
                f"(max {self.max_windows_per_record} windows/record)  [eager]"
            )
        except MemoryError as e:
            self._windows = None
            print(f"  -> Falling back to lazy/memmap loading ({e})")
            print(
                f"Created dataset: {len(self.labels)} windows "
                f"({np.sum(self.labels == 1)} AF, {np.sum(self.labels == 0)} SR) "
                f"from {len(self.file_paths)} records "
                f"(max {self.max_windows_per_record} windows/record)  [lazy/memmap]"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self._windows is not None:
            # Fast path: already in RAM
            chunk = self._windows[idx]
        else:
            # Lazy path: memmap on demand
            dat_path, start, window_size, downsample_factor = self._items[idx]
            mm = np.memmap(dat_path, dtype=np.float32, mode='r')
            if downsample_factor > 1:
                raw_start = start * downsample_factor
                chunk = np.array(
                    mm[raw_start: raw_start + window_size * downsample_factor: downsample_factor],
                    dtype=np.float32,
                )
            else:
                chunk = np.array(mm[start: start + window_size], dtype=np.float32)
            del mm
            std = chunk.std()
            chunk = (chunk - chunk.mean()) / (std + 1e-8)

        x = torch.from_numpy(chunk.copy()).unsqueeze(0)  # shape: [1, L]
        y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)  # [1]
        rec_idx = int(self.record_index[idx])

        if self.augment:
            # 1. Gaussian noise
            if torch.rand(1) < 0.75:  # more frequent
                noise_level = torch.rand(1).item() * 0.08 + 0.02  # 2% to 10% of std
                noise = torch.randn_like(x) * noise_level
                x = x + noise

            # 2. Random sign flip (invert polarity)
            if torch.rand(1) < 0.10:
                x = -x

        # Normalize again after augmentation (important!)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std

        return x, y, rec_idx
