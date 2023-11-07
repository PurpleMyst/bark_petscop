from datetime import timedelta
from pathlib import Path
from subprocess import run
from time import perf_counter

import argh
import nltk
import numpy as np
import scipy
import srt
from bark import SAMPLE_RATE
from bark.api import semantic_to_waveform
from bark.generation import generate_text_semantic, preload_models
from joblib import Memory
from tqdm import main, tqdm
from win10toast import ToastNotifier

MEMORY = Memory("joblib_cache", verbose=0)

SILENCE_DURATION = 0.1


@MEMORY.cache
def generate_sentence(sentence: str):
    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_7"

    semantic_tokens = generate_text_semantic(
        sentence,
        history_prompt=SPEAKER,
        temp=GEN_TEMP,
        min_eos_p=0.05,  # this controls how likely the generation is to end
    )
    audio_array = semantic_to_waveform(
        semantic_tokens,
        history_prompt=SPEAKER,
    )
    return audio_array


@argh.arg(
    "--encode-slow", default=False, help="Encode the video slowly, but with a smaller file size."
)
def main(encode_slow: bool = False) -> None:
    preload_models()

    nltk.download("punkt")

    sentences = nltk.sent_tokenize(Path("text.txt").read_text(encoding="utf-8"))
    silence = np.zeros(int(SILENCE_DURATION * SAMPLE_RATE))

    # Raw WAV data and srt.Subtitle objects.
    audio_arrays = []
    subtitles = []

    # Timestamp so far.
    timestamp = timedelta(seconds=0)

    # Generate audio for each sentence, stopping if the user presses Ctrl+C.
    try:
        padding = len(str(len(sentences)))
        for i, sentence in enumerate(sentences):
            tqdm.write(f"[{i + 1:{padding}}/{len(sentences)}] {sentence!r}")
            start = perf_counter()
            sentence_audio = generate_sentence(sentence)
            duration = timedelta(seconds=len(sentence_audio) / SAMPLE_RATE)
            gen_time = perf_counter() - start
            ratio = gen_time / duration.total_seconds()
            if ratio > 0.05:
                # If ratio is too low, we'd probably already cached it.
                tqdm.write(
                    f"Generated {duration.total_seconds():.2f}s of audio in {gen_time:.2f}s"
                    f" ({ratio:.2f}x realtime)"
                )
            audio_arrays.append(sentence_audio)
            audio_arrays.append(silence)
            subtitles.append(
                srt.Subtitle(
                    index=i,
                    start=timestamp,
                    end=timestamp + duration,
                    content=srt.make_legal_content(sentence),
                )
            )
            timestamp += duration + timedelta(seconds=SILENCE_DURATION)
    except KeyboardInterrupt:
        pass

    # Write out the final WAV and SRT files.
    try:
        audio_array = np.concatenate(audio_arrays)
        scipy.io.wavfile.write("output.wav", rate=SAMPLE_RATE, data=audio_array)
        Path("petscop.srt").write_text(srt.compose(subtitles), encoding="utf-8")

        # Use ffmpeg to combine the WAV and SRT files into an MP4, along with a pretty picture.
        cmd = (
            "ffmpeg",
            "-hide_banner",
            "-threads",
            "8",
            "-loop",
            "1",
            "-y",
            "-i",
            "petscop.jpg",
            "-i",
            "output.wav",
            "-i",
            "petscop.srt",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "h264_nvenc",
            "-shortest",
            "-c:s",
            "copy",
            "petscop.mkv",
        )
        run(
            cmd,
            check=True,
        )
    finally:
        Path("output.wav").unlink(missing_ok=True)
        Path("petscop.srt").unlink(missing_ok=True)


if __name__ == "__main__":
    argh.dispatch_command(main)
