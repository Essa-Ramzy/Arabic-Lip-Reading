import torch
import editdistance
import cv2
import math
import random
import pandas as pd
import numpy as np
from PIL import Image
import kornia.augmentation as K
from torchvision import transforms

MEAN = 0.421
STD = 0.165

def set_normalization_params(mean, std):
    """
    Update the global MEAN and STD values used for normalization.
    
    Args:
        mean (float): New mean value
        std (float): New standard deviation value
    """
    global MEAN, STD
    MEAN = mean
    STD = std
    print(f"Updated normalization parameters: MEAN={MEAN}, STD={STD}")

def pad_packed_collate(batch):
    """Pads data and labels with different lengths in the same batch
    """
    data_list, input_lengths, labels_list, label_lengths = zip(*batch)
    c, max_len, h, w = max(data_list, key=lambda x: x.shape[1]).shape

    data = torch.zeros((len(data_list), c, max_len, h, w))
    
    # Only copy up to the actual sequence length
    for idx in range(len(data)):
        data[idx, :, :input_lengths[idx], :, :] = data_list[idx][:, :input_lengths[idx], :, :]
    
    # Flatten labels for CTC loss
    labels_flat = []
    for label_seq in labels_list:
        labels_flat.extend(label_seq)
    labels_flat = torch.LongTensor(labels_flat)
    
    # Convert lengths to tensor
    input_lengths = torch.LongTensor(input_lengths)
    label_lengths = torch.LongTensor(label_lengths)
    return data, input_lengths, labels_flat, label_lengths

def seed_worker(worker_id):
    """
    Seeds all random number generators for a DataLoader worker.
    """
    # Get the initial seed from the main process to ensure that the sequence of
    # worker seeds is the same across different runs.
    worker_seed = torch.initial_seed() % 2**32
    final_seed = worker_seed + worker_id

    # Seed all relevant libraries
    np.random.seed(final_seed)
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    # Seed CUDA RNGs for this worker, if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(final_seed)

def indices_to_text(indices, idx2char):
    """
    Converts a list of indices to text using the reverse vocabulary mapping.
    """
    try:
        n_vocab = len(idx2char)
        return ''.join([idx2char[i] for i in indices if i < n_vocab])
    except UnicodeEncodeError:
        # Handle encoding issues in Windows console
        # Return a safe representation that won't cause encoding errors
        safe_text = []
        for i in indices:
            char = idx2char.get(i, '')
            try:
                # Test if character can be encoded
                char.encode('cp1252')
                safe_text.append(char)
            except UnicodeEncodeError:
                # Replace with a placeholder for characters that can't be displayed
                safe_text.append(f"[{i}]")
        return ''.join(safe_text)

def compute_cer(reference_indices, hypothesis_indices):
    """
    Computes Character Error Rate (CER) directly using token indices.
    Takes raw token indices from our vocabulary (class_mapping.txt).
    Returns a tuple of (CER, edit_distance).
    """
    # Use the indices directly: each index is one token
    ref_tokens = reference_indices
    hyp_tokens = hypothesis_indices
    # Calculate edit distance
    edit_distance = editdistance.eval(ref_tokens, hyp_tokens)
    # Calculate CER (avoid division by zero)
    cer = edit_distance / max(len(ref_tokens), 1)
    return cer, edit_distance

def indices_to_text_word(indices, idx2token):
    """
    Converts a list of word indices to a space-separated string using the reverse vocabulary mapping.
    """
    return ' '.join([idx2token.get(i, '') for i in indices])


def compute_wer(reference_indices, hypothesis_indices):
    """
    Computes Word Error Rate (WER) directly using token indices.
    Returns a tuple of (WER, edit_distance).
    """
    # Calculate edit distance between reference and hypothesis tokens
    edit_distance = editdistance.eval(reference_indices, hypothesis_indices)
    wer = edit_distance / max(len(reference_indices), 1)
    return wer, edit_distance

def extract_label(file, with_spaces=False, with_diaritics=True):
    label = []
    diacritics = {
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u06E2',  # Small High meem
    }

    sentence = pd.read_csv(file)
    for i, word in enumerate(sentence.word):
        if with_spaces and i:
            label.append(' ')
        for char in word:
            if char not in diacritics:
                label.append(char)
            elif with_diaritics:
                label[-1] += char

    return label

# Video augmentation class using Kornia VideoSequential with assertions
class VideoAugmentation:
    def __init__(self, is_train=True, crop_size=(88, 88)):
        if is_train:
            self.aug = K.VideoSequential(
                K.RandomCrop(crop_size, p=1.0),
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(brightness=0.4, contrast=0.4, p=0.6),
                K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.6),
                K.RandomAffine(degrees=10, translate=0.1, scale=(0.9, 1.1), p=0.6),
                data_format="BCTHW",
                same_on_frame=True,
            )
        else:
            self.aug = K.VideoSequential(
                K.CenterCrop(crop_size, p=1.0),
                data_format="BCTHW",
                same_on_frame=True,
            )
        self.crop_size = crop_size
        self.is_train = is_train
        self.time_mask_prob = 0.7 if is_train else 0.0
        self.temporal_augment_prob = 0.6 if is_train else 0.0

    def adaptive_time_mask(self, x, mask_window=4, mask_stride=40):
        """Apply adaptive time masking to video tensor"""
        # x shape is (C, T, H, W)
        cloned = x.clone()
        C, T, H, W = cloned.shape
        # Calculate how many masks to apply
        n_mask = int((T + mask_stride - 0.1) // mask_stride)

        if n_mask < 1:
            return cloned

        # This is a standard temporal masking implementation.
        for _ in range(n_mask):
            # Pick a mask length up to `window` frames
            mask_len = torch.randint(0, mask_window, (1,)).item()
            if T > mask_len > 0:
                # Pick a start point
                t_start = random.randrange(0, T - mask_len)
                cloned[:, t_start : t_start + mask_len, :, :] = 0
        return cloned

    def temporal_jitter(self, video, max_shift=2):
        """Shift the video slightly in time by +/- frames"""
        C, T, H, W = video.shape
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return video
        elif shift > 0:
            pad = video[:, :1, :, :].repeat(1, shift, 1, 1)
            return torch.cat([pad, video[:, :-shift, :, :]], dim=1)
        else:
            pad = video[:, -1:, :, :].repeat(1, -shift, 1, 1)
            return torch.cat([video[:, -shift:, :, :], pad], dim=1)
            
    def __call__(self, pil_frames):
        # Convert list of PIL images to tensor sequence
        frame_tensors = [transforms.ToTensor()(img) for img in pil_frames]
        video = torch.stack(frame_tensors, dim=0)      # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)              # (C, T, H, W)
        video_batch = video.unsqueeze(0)               # (1, C, T, H, W)

        # Apply Kornia augmentations
        augmented = self.aug(video_batch)              # (1, C, T, H, W)
        augmented = augmented.squeeze(0)               # (C, T, H, W)

        if self.is_train and torch.rand(1).item() < self.time_mask_prob:
            augmented = self.adaptive_time_mask(augmented)

        if self.is_train and torch.rand(1).item() < self.temporal_augment_prob:
            augmented = self.temporal_jitter(augmented)
            
        # Assertions for shape and validity
        C, T, H, W = augmented.shape
        assert C == 1, f"Expected channel=1, got {C}"
        assert (H, W) == self.crop_size, f"Expected spatial size {self.crop_size}, got {(H,W)}"
        assert not torch.isnan(augmented).any(), "NaNs in augmented clip!"
        assert not torch.isinf(augmented).any(), "Infs in augmented clip!"
        # Normalize channels
        augmented = (augmented - MEAN) / STD
        return augmented

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, label_paths, mapped_tokens, with_spaces=False, with_diaritics=True, transform=None):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.mapped_tokens = mapped_tokens
        self.with_spaces = with_spaces
        self.with_diaritics = with_diaritics
        self.transform = transform
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label_path = self.label_paths[index]
        frames = self.load_frames(video_path=video_path)
        label = torch.tensor(list(map(lambda x: self.mapped_tokens[x], extract_label(label_path, with_spaces=self.with_spaces, with_diaritics=self.with_diaritics))))
        input_length = torch.tensor(frames.size(1), dtype=torch.long)
        label_length = torch.tensor(len(label), dtype=torch.long)
        return frames, input_length, label, label_length
    
    def load_frames(self, video_path):
        frames = []
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frames):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_pil = Image.fromarray(frame, 'L')
                frames.append(frame_pil)

        if self.transform is not None:
            # Apply video-level transformation
            video = self.transform(frames)
        else:
            # Fallback: per-frame ToTensor + Normalize
            frame_tensors = []
            for img in frames:
                t = transforms.ToTensor()(img)
                t = transforms.Normalize(mean=[MEAN], std=[STD])(t)
                frame_tensors.append(t)
            video = torch.stack(frame_tensors).permute(1, 0, 2, 3)
        return video

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self._step_count / self.warmup_steps * base_lr for base_lr in self.base_lrs]
        decay_steps = self.total_steps - self.warmup_steps
        cos_val = math.cos(math.pi * (self._step_count - self.warmup_steps) / decay_steps)
        return [0.5 * base_lr * (1 + cos_val) for base_lr in self.base_lrs]
