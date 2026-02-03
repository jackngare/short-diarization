# short-diarization

Lightweight audio diarization / conservative transcription example using Google GenAI (Gemini) for highly conservative, anti-hallucination transcription. The script pre-analyzes audio to detect meaningful speech and only sends audio to Gemini when there is a reasonable chance of producing a reliable transcript.

> Note: This repository contains an example integration. You must supply your own GCP project, credentials, and audio files.

## Features
- Pre-analyzes WAV audio for RMS energy and speech-duration estimates to avoid transcribing silence/noise.
- Sends audio to a Gemini model with a strict, deterministic prompt to minimize hallucination.
- Validates Gemini JSON output and filters low-confidence entries.
- Prints a human-friendly, time-stamped transcript and a JSON-compatible representation.

## Requirements
- Python 3.8+
- Google GenAI client libraries (the code uses `google-genai` / `google.genai`)
- NumPy
- A WAV audio file (PCM). Script handles mono and stereo.

Recommended Python packages:
- google-genai (or the appropriate client library available for your environment)
- numpy

Install with pip:
```
pip install numpy google-genai
```

## Configuration

1. Enable Vertex AI / GenAI in your Google Cloud project and ensure that the GenAI APIs are available to your project.
2. Create and download a service account key and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to the JSON key file:
```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```
3. Open `diarize.py` and set:
   - `PROJECT_ID` to your GCP project id
   - `LOCATION` if different from the default (`us-central1`)
   - `audio_path` to the audio file you want to transcribe (or modify the script to accept CLI args)

You can also change the model by adjusting the `model_name` passed into `GeminiSpeechProcessor` (default: `"gemini-2.5-flash"` in the example).

## Usage

Edit `diarize.py` to set:
- `PROJECT_ID = 'your-project-id'`
- `audio_path = 'path/to/your_audio.wav'`

Then run:
```
python diarize.py
```

Output:
- The script will first print audio analysis (duration, RMS energy, speech duration, silence ratio).
- If no meaningful speech is detected, transcription is skipped to avoid hallucination.
- If audio is sent to Gemini, the script prints processing time and a formatted transcript of validated, high-confidence entries. It also attempts to parse and pretty-print the JSON response returned by the model.

## How it works (high level)
1. analyze_audio_content(): Reads the WAV file, computes RMS energy and speech duration using 100ms windows, and decides whether meaningful speech exists based on configurable thresholds.
2. process(): If speech is detected, reads the bytes of the WAV and calls Gemini via the GenAI client with a conservative, deterministic prompt (JSON response expected).
3. _validate_and_print_transcript(): Parses the JSON response, filters entries with confidence < 0.7 (configurable), prints warnings if the transcript seems too long for the detected speech, and prints final validated entries.

## Configuration parameters to tune
- `silence_threshold` (default 0.005) — RMS threshold to consider a 100ms window "speech".
- `min_speech_duration` (default 0.2s) — minimum total speech duration to attempt transcription.
- `confidence` filter threshold in `_validate_and_print_transcript()` (default 0.7).
- `window_size` in `analyze_audio_content()` — currently fixed to 100ms; you can reduce or increase granularity.

## Supported audio formats
- The script expects PCM WAV files. It supports sample widths of 1, 2 or 4 bytes and mono/stereo audio. Non-PCM or compressed WAVs may not work correctly.

## Security, costs, and best practices
- GenAI/Vertex AI calls will incur costs. Test with short audio and be mindful of model selection and quotas.
- The script is intentionally conservative to avoid hallucination. For higher recall, you can relax thresholds but risk more hallucinated output.
- Keep your service account JSON secure and do not commit it to source control.

## Limitations
- This example is not a complete diarization system (no speaker embedding/clustering). It relies on the LLM to label speakers if it can confidently identify multiple voices.
- The script expects the model to return JSON in the specified schema. If the model returns malformed JSON, the raw text is printed.
- The current usage model requires editing variables in the file. Consider adding CLI arg support for production use.

## Contributing
Contributions are welcome. If you want to:
- Add CLI argument parsing
- Support more audio formats (e.g., FLAC, MP3) via decoding
- Add speaker embedding-based diarization (e.g., pyannote, spectral clustering)
- Add unit tests

Please open an issue or PR with proposed changes.

## License
Apache License 2.0 — see LICENSE file.

## References
- Google GenAI / Vertex AI documentation
- Your project's internal notes on hallucination mitigation and safety
