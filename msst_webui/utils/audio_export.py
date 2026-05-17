import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf


def _require_ffmpeg():
	ffmpeg = shutil.which("ffmpeg")
	if not ffmpeg:
		raise RuntimeError("MP3 output requires ffmpeg on PATH. Install ffmpeg or choose wav/flac output.")
	return ffmpeg


def _audio_for_mp3(audio):
	audio = np.asarray(audio)
	if np.issubdtype(audio.dtype, np.floating):
		return np.clip(audio, -1.0, 1.0)
	return audio


def export_mp3(audio, sr, file, bitrate="320k"):
	ffmpeg = _require_ffmpeg()
	audio = _audio_for_mp3(audio)
	temp_file = None
	try:
		with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
			temp_file = handle.name
		sf.write(temp_file, audio, sr, subtype="PCM_16", format="WAV")
		command = [
			ffmpeg,
			"-y",
			"-hide_banner",
			"-loglevel",
			"error",
			"-i",
			temp_file,
			"-vn",
			"-b:a",
			str(bitrate),
			file,
		]
		subprocess.run(command, check=True)
	finally:
		if temp_file and os.path.exists(temp_file):
			os.remove(temp_file)


def save_audio_file(audio, sr, output_format, file_name, store_dir, wav_bit_depth="FLOAT", flac_bit_depth="PCM_24", mp3_bit_rate="320k"):
	output_format = output_format.lower()
	if output_format == "flac":
		file = os.path.join(store_dir, file_name + ".flac")
		sf.write(file, audio, sr, subtype=flac_bit_depth)
	elif output_format == "mp3":
		file = os.path.join(store_dir, file_name + ".mp3")
		export_mp3(audio, sr, file, bitrate=mp3_bit_rate)
	else:
		file = os.path.join(store_dir, file_name + ".wav")
		sf.write(file, audio, sr, subtype=wav_bit_depth)
	return file
