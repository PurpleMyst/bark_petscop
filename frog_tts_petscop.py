from contextlib import redirect_stdout
from datetime import timedelta
from io import StringIO
from pathlib import Path
from subprocess import run
from time import perf_counter
from uuid import uuid4
from math import ceil

import argh
import nltk
import numpy as np
import scipy
import srt
from joblib import Memory
from rich import print
from tqdm import tqdm
from TTS.api import TTS

MEMORY = Memory("joblib_cache", verbose=0)
tts = TTS("tts_models/en/jenny/jenny").to("cuda")

SILENCE_DURATION = 0.1


@MEMORY.cache
def generate_sentence(sentence: str):
    start = perf_counter()
    with redirect_stdout(StringIO()):
        audio_array = tts.tts(sentence)
    gen_time = perf_counter() - start
    return audio_array, gen_time


def batched(iterable, n):
    from itertools import islice

    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            break
        yield batch


def main() -> None:
    nltk.download("punkt")

    all_sentences = nltk.sent_tokenize(Path("text.txt").read_text(encoding="utf-8"))
    prefix = f"petscop-{uuid4()}"
    files = []

    assert tts.synthesizer is not None, "TTS model does not have a synthesizer?"
    sample_rate = tts.synthesizer.output_sample_rate

    print("\033c",end="")  # Clear console

    # Break the sentences into batches of 500, and generate audio for each batch.
    # This is done to avoid WAV limitations (and also because otherwise we'd use a lot of RAM)
    # We can just concatenate it all together later.
    for j, sentences in enumerate(tqdm(batched(all_sentences, 500), desc="Batches", unit="batch", total=ceil(len(all_sentences) / 500))):
        audio_arrays = []
        subtitles = []

        timestamp = timedelta(seconds=0)

        # padding = len(str(len(sentences)))
        for i, sentence in enumerate(tqdm(sentences, desc="Sentences", unit="sentence"), start=1):
            # h, ms = divmod(timestamp.total_seconds(), 60 * 60)
            # m, s = divmod(ms, 60)
            # print(
            #     rf"[yellow]\[{i + 1:{padding}}/{len(sentences)} - {h:02.0f}:{m:02.0f}:{s:02.0f}]"
            #     rf" {sentence!r}"
            # )

            sentence_audio, _gen_time = generate_sentence(sentence)
            duration = timedelta(seconds=len(sentence_audio) / sample_rate)
            # ratio = gen_time / duration.total_seconds()
            # print(
            #     f"\tGenerated [red]{duration.total_seconds():.2f}s[/] of audio in"
            #     f" [red]{gen_time:.2f}s[/] ([bold cyan]{ratio:.2f}x realtime[/])"
            # )

            audio_arrays.append(sentence_audio)
            silence = np.zeros(int(SILENCE_DURATION * sample_rate))
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

        output = f"{prefix}-{j:02}.mkv"
        with tqdm.external_write_mode():
            print(f"Writing output to [bold cyan]{output}[/]")
        files.append(output)

        try:
            audio_array = np.concatenate(audio_arrays)
            scipy.io.wavfile.write("output.wav", rate=sample_rate, data=audio_array)
            Path("petscop.srt").write_text(srt.compose(subtitles), encoding="utf-8")

            # Use ffmpeg to combine the WAV and SRT files into a video file, along with a pretty picture.
            cmd = (
                "ffmpeg",
                "-hide_banner",
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
                output,
            )
            run(cmd, check=True)
        finally:
            Path("output.wav").unlink(missing_ok=True)
            Path("petscop.srt").unlink(missing_ok=True)

    # At this point, we have a bunch of video files. We can use ffmpeg to concatenate them all together.
    print(f"Concatenating all videos into [bold cyan]{prefix}.mkv[/]")
    Path(f"{prefix}-files.txt").write_text("\n".join(f"file '{f}'" for f in files))
    try:
        cmd = (
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            f"{prefix}-files.txt",
            "-c",
            "copy",
            f"{prefix}.mkv",
        )
        run(cmd, check=True)
    finally:
        Path(f"{prefix}-files.txt").unlink(missing_ok=True)

    print(f"Done! :frog:")


if __name__ == "__main__":
    argh.dispatch_command(main)
