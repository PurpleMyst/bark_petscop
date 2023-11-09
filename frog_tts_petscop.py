from contextlib import redirect_stdout
from datetime import timedelta
from io import StringIO
from itertools import chain
from math import ceil
from os import get_terminal_size
from pathlib import Path
from random import choice
from subprocess import run
from time import perf_counter
from uuid import uuid4

import argh
import nltk
import numpy as np
import scipy
import srt
from joblib import Memory
from rich.console import Console
from TTS.api import TTS

print = Console(highlight=False).print

MEMORY = Memory("joblib_cache", verbose=0)

BATCH_SIZE = 500
SILENCE_DURATION = 0.1
EMPTY_STRING_DURATION = 2 * SILENCE_DURATION

VIDEO_ENCODER = "h264_nvenc"  # h264_nvenc is faster than libx264 but less configurabile
AUDIO_ENCODER = "aac"  # libopus is faster & better than aac but less compatible


@MEMORY.cache(ignore=["tts"])
def generate_sentence(tts, sentence: str, *, language: str | None, speaker: str | None):
    start = perf_counter()
    with redirect_stdout(StringIO()):
        audio_array = tts.tts(sentence, language=language, speaker_wav=speaker)
    gen_time = perf_counter() - start
    return audio_array, gen_time


# taken from the itertools 3.12 docs
def batched(iterable, n):
    from itertools import islice

    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            break
        yield batch


def convert_batch(tts, batch, *, output, image, language, speaker):
    sample_rate = tts.synthesizer.output_sample_rate

    audio_arrays = []
    subtitles = []

    timestamp = timedelta(seconds=0)
    ki = None

    try:
        padding = len(str(len(batch)))
        for i, text in enumerate(batch, start=1):
            h, ms = divmod(timestamp.total_seconds(), 60 * 60)
            m, s = divmod(ms, 60)
            print(
                rf"[yellow]\[{i:{padding}}/{len(batch)} - {h:02.0f}:{m:02.0f}:{s:02.0f}]"
                rf" {text!r}"
            )

            if not text.strip():
                silence = np.zeros(int(EMPTY_STRING_DURATION * sample_rate))
                audio_arrays.append(silence)
                timestamp += timedelta(seconds=EMPTY_STRING_DURATION)
                continue

            sentence_audio, gen_time = generate_sentence(
                tts, text, language=language, speaker=speaker
            )
            duration = timedelta(seconds=len(sentence_audio) / sample_rate)
            ratio = gen_time / duration.total_seconds()
            print(
                f"\tGenerated [red]{duration.total_seconds():.2f}s[/] of audio in"
                f" [red]{gen_time:.2f}s[/] ([bold cyan]{ratio:.2f}x realtime[/])"
            )

            audio_arrays.append(sentence_audio)
            silence = np.zeros(int(SILENCE_DURATION * sample_rate))
            audio_arrays.append(silence)

            subtitles.append(
                srt.Subtitle(
                    index=i,
                    start=timestamp,
                    end=timestamp + duration,
                    content=srt.make_legal_content(text),
                )
            )

            timestamp += duration + timedelta(seconds=SILENCE_DURATION)
    except KeyboardInterrupt as e:
        ki = e

    print(f"Writing output to [bold cyan]{output}[/]")

    try:
        audio_array = np.concatenate(audio_arrays)
        scipy.io.wavfile.write("output.wav", rate=sample_rate, data=audio_array)
        Path("petscop.srt").write_text(srt.compose(subtitles), encoding="utf-8")

        # Use ffmpeg to combine the WAV and SRT files into a video file, along with a pretty picture.
        # ---
        # Pad the image to the nearest even width/height, otherwise we get an error in libx264.
        img_filter = "[0:v]pad=ceil(iw/2)*2:ceil(ih/2)*2[img]"
        waves_filter = (
            "[1:a]showwaves=mode=line:s=xga:colors=Blue@0.5:scale=sqrt,format=yuva420p[waves]"
        )
        overlay_filter = "[img][waves]overlay=x=(W-w)/2:y=(H-h)/2[out]"
        filter_complex = f"{img_filter};{waves_filter};{overlay_filter}"
        # fmt: off
        cmd = (
            "ffmpeg",
            "-hide_banner",
            "-loop", "1",
            "-y",
            "-i", image,
            "-i", "output.wav",
            "-i", "petscop.srt",
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "1:a",
            "-map", "2:s",
            "-c:v", VIDEO_ENCODER,
            "-crf", "28",
            "-shortest",
            "-c:s", "mov_text",
            "-c:a", AUDIO_ENCODER,
            output,
        )
        run(cmd, check=True)
        # fmt: on
    finally:
        Path("output.wav").unlink(missing_ok=True)
        Path("petscop.srt").unlink(missing_ok=True)

    if ki is not None:
        raise ki

    return output


def concatenate(files, output):
    files_file = Path(output).with_suffix(".files.txt")
    files_file.write_text("\n".join(f"file '{Path(f).resolve()}'" for f in files))
    try:
        # fmt: off
        cmd = (
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", files_file,
            "-c", "copy",
            "-c:s", "copy",
            output,
        )
        run(cmd, check=True)
        # fmt: on
    finally:
        files_file.unlink(missing_ok=True)


def break_sentences(text: str) -> list[str]:
    return list(chain.from_iterable(sent.split("\n") for sent in nltk.sent_tokenize(text)))


@argh.arg("input_filepath", type=str, help="The plain text file to convert to speech.")
@argh.arg(
    "-l",
    "--language",
    type=str,
    help="The language to use for the text-to-speech. Defaults to None as the default model is not multi-language.",
)
@argh.arg(
    "-s",
    "--speaker",
    type=str,
    help="A WAV file to use as the speaker's voice. If left unspecified, Jenny will be used (she's faster than cloning a speaker)",
)
def main(input_filepath: str, language: str | None = None, speaker: str | None = None) -> None:
    nltk.download("punkt")

    if speaker is None:
        tts = TTS("tts_models/en/jenny/jenny")
    else:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts = tts.to("cuda")

    Path("output").mkdir(exist_ok=True)

    # Split the text into sentences, and then split each sentence into lines to ensure that the
    # subtitles don't get too long.
    all_sentences = break_sentences(Path(input_filepath).read_text(encoding="utf-8"))
    prefix = f"output/{Path(input_filepath).stem}-{uuid4()}"
    image = choice(list(Path("images").glob("*")))
    files = []

    print("\033c", end="")  # Clear console

    # Break the sentences into batches, and generate audio for each batch.
    # This is done to avoid WAV limitations (and also because otherwise we'd use a lot of RAM)
    # We can just concatenate it all together later.
    try:
        try:
            for j, sentences in enumerate(batched(all_sentences, BATCH_SIZE), start=1):
                print(
                    "[bold magenta]Generating audio for batch"
                    f" {j}/{ceil(len(all_sentences) / BATCH_SIZE)}"
                )
                print(f"[magenta]{'-' * get_terminal_size().columns}[/]")
                output = f"{prefix}-{j:02}.mp4"
                files.append(output)  # so it gets deleted if something goes wrong
                convert_batch(
                    tts, sentences, output=output, image=image, language=language, speaker=speaker
                )
        except KeyboardInterrupt:
            pass

        print(f"Concatenating all videos into [bold cyan]{prefix}.mp4[/]")
        concatenate(files, f"{prefix}.mp4")
    finally:
        for file in files:
            Path(file).unlink(missing_ok=True)

    print("Done! :frog:")


if __name__ == "__main__":
    argh.dispatch_command(main)
