import torch
import torchvision
import torchaudio
from torchvision import transforms
from torchvision.io import read_video
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os
import argparse
from tqdm import tqdm
import decord
from decord import VideoReader, cpu

# ====== Args ======
parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, default='/data/stonekab/crema-d-mirror/VideoFlash')
parser.add_argument('--audio_dir', type=str, default='/data/stonekab/crema-d-mirror/AudioWAV')
parser.add_argument('--batch_size', type=int, default=8)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = f"/data/stonekab/cached_cremad_feats"
os.makedirs(save_dir, exist_ok=True)

# ====== Video encoder ======
video_encoder = torchvision.models.video.r3d_18(pretrained=True).to(device)
video_encoder.eval()
video_preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
])

# ====== Audio encoder ======
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
audio_encoder.eval()

# ====== Helper ======
def parse_filename(filename):
    parts = filename.split("_")
    subject_id = parts[0]
    sentence_id = parts[1]
    feeling = parts[2]
    return subject_id, sentence_id, feeling

# ====== Main processing ======
all_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".flv")])
all_data = []

for i in tqdm(range(0, len(all_files), args.batch_size)):
    batch_files = all_files[i:i + args.batch_size]

    for filename in batch_files:
        subj_id, sent_id, feeling = parse_filename(filename)

        # --- Video ---
        video_path = os.path.join(args.video_dir, filename)
        vr = VideoReader(video_path, ctx=cpu(0))
        video_frames = vr.get_batch(range(0, len(vr)))
        video_frames = torch.from_numpy(video_frames.asnumpy())  # (T, H, W, C)
        video_frames = video_frames[:16]
        video_frames = video_frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

        video_preprocess_tensor = transforms.Compose([
                    transforms.Resize((112, 112)),  # Already expects tensor
                    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                        std=[0.22803, 0.22145, 0.216989]),
        ])
        video_frames = torch.stack([video_preprocess_tensor(f) for f in video_frames])
        video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)  # (1, C, T, H, W)


        with torch.no_grad():
            vid_feat = video_encoder(video_frames).squeeze(0).cpu()  # [D]

        # --- Audio ---
        audio_name = filename.replace(".flv", ".wav")
        audio_path = os.path.join(args.audio_dir, audio_name)
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        inputs = audio_processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_outputs = audio_encoder(**inputs)
            aud_feat = audio_outputs.last_hidden_state.mean(dim=1).cpu()  # [D]

        # --- Save ---
        all_data.append({
            "index": i,
            "subject_id": subj_id,
            "sentence_id": sent_id,
            "feeling": feeling,
            "video_feat": vid_feat,
            "audio_feat": aud_feat
        })

    torch.cuda.empty_cache()

save_path = os.path.join(save_dir, f"cached_cremad_feats.pt")
torch.save(all_data, save_path)
print(f"Saved {len(all_data)} items to {save_path}")
