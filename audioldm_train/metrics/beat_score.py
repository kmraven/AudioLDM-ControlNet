import librosa
import numpy as np
import torch
import scipy.signal as scisignal


def beat_detect(x, sr):
    """
    from LORIS: https://github.com/OpenGVLab/LORIS/blob/443e4b992a1dc32ab594373012e9fef9928ce404/beats_scores.py#L16
    - time resolution for the music beat detection is modified to 5 Hz (following a setting of textual inversion)
    """
    if x.ndim > 1:
        x = librosa.to_mono(x)
    onsets = librosa.onset.onset_detect(
        y=x,
        sr=sr,
        wait=1,
        delta=0.2,
        pre_avg=3,
        post_avg=3,
        pre_max=3,
        post_max=3,
        units="time",
    )
    time_resolution = 5  # 5 Hz
    n = np.ceil(len(x) / sr * time_resolution)
    beats = [0] * int(n)
    for time in onsets:
        beats[int(np.trunc(time * time_resolution))] = 1
    return beats


def beat_scores(gt, syn):
    """
    from LORIS: https://github.com/OpenGVLab/LORIS/blob/443e4b992a1dc32ab594373012e9fef9928ce404/beats_scores.py#L24
    """
    gt = gt[: len(syn)]
    assert len(gt) == len(syn)
    total_beats = sum(gt)
    cover_beats = sum(syn)
    hit_beats = 0
    for i in range(len(gt)):
        if gt[i] == 1 and gt[i] == syn[i]:
            hit_beats += 1
    hit_rate = hit_beats / total_beats if total_beats else 0
    cover_rate = hit_beats / cover_beats if cover_beats else 0
    return cover_rate, hit_rate


def f1_score(cover_rate, hit_rate):
    if (cover_rate + hit_rate) == 0:
        return 0
    return 2 * (cover_rate * hit_rate) / (cover_rate + hit_rate)


def estimate_bpm(x, sr, aggregate=np.mean):
    """
    tempo_difference is originally implemented since authors of Textual_inversion doesn't provide it
    """
    if x.ndim > 1:
        x = librosa.to_mono(x)
    onset_env = librosa.onset.onset_strength(y=x, sr=sr)
    if onset_env is None or np.allclose(onset_env.sum(), 0.0):
        return 0.0
    # Global tempo (BPM): aggregate with median for robustness
    bpm_arr = librosa.beat.tempo(
        onset_envelope=onset_env, sr=sr, aggregate=aggregate
    )
    # librosa returns a 1-element array for global tempo
    bpm = float(np.squeeze(bpm_arr))
    # Guard against NaN/Inf
    if not np.isfinite(bpm):
        return 0.0
    return bpm


def tempo_difference(gt_audio, syn_audio, sr):
    bpm_gt = estimate_bpm(gt_audio, sr=sr)
    bpm_syn = estimate_bpm(syn_audio, sr=sr)
    return abs(bpm_syn - bpm_gt)


def music_peak_onehot(music, sr):
    """
    from AI Choreographer: https://github.com/liruilong940607/mint/blob/d40065ab1d2021768ce8f474e1affa7eb2a121a4/tools/preprocessing.py#L99
    """
    MOTION_FPS = 60
    HOP_LENGTH = 512
    target_sr = int(MOTION_FPS * HOP_LENGTH)  # 30720
    music = librosa.resample(music, orig_sr=sr, target_sr=target_sr)
    envelope = librosa.onset.onset_strength(y=music, sr=target_sr)  # (seq_len,)
    _, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope, sr=target_sr, hop_length=HOP_LENGTH, tightness=100
    )  # not specifiy the start_bpm since the bpm of generated music is unknown
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)
    return beat_onehot


def motion_peak_onehot(joints):
    """
    from AI Choreographer: https://github.com/liruilong940607/mint/blob/d40065ab1d2021768ce8f474e1affa7eb2a121a4/tools/calculate_beat_scores.py#L89

    Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - peak_onhot: motion beats.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # # Second-derivative of the velocity shows the energy of the beats
    # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
    # # optimize peaks
    # peak_onehot[peak_energy<0.001] = 0
    return peak_onehot


def alignment_score(music_beats, motion_beats, sigma=3):
    """
    from AI Choreographer: https://github.com/liruilong940607/mint/blob/d40065ab1d2021768ce8f474e1affa7eb2a121a4/tools/calculate_beat_scores.py#L114
    
    Calculate alignment score between music and motion.
    """
    assert len(music_beats) == len(motion_beats)

    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all)


def calculate_beat_score(gt_motion, gt_audio, gen_audio, sr):
    if isinstance(gt_audio, torch.Tensor) == True:
        gt_audio = gt_audio.cpu().numpy()
    if isinstance(gen_audio, torch.Tensor) == True:
        gen_audio = gen_audio.cpu().numpy()
    gt_music_beats = beat_detect(gt_audio, sr)
    gen_music_beats = beat_detect(gen_audio, sr)
    cover_rate, hit_rate = beat_scores(gt_music_beats, gen_music_beats)
    f1 = f1_score(cover_rate, hit_rate)
    td = tempo_difference(gt_audio, gen_audio, sr=sr)
    bas = alignment_score(
        music_peak_onehot(gen_audio, sr),
        motion_peak_onehot(gt_motion),
        sigma=3
    )
    return cover_rate, hit_rate, f1, td, bas