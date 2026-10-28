# TrainerTwin take-home: voice interview coach

A cascaded voice agent (speech-to-text → LLM → text-to-speech) built with Pipecat. "Maya", a friendly interview coach, runs a short mock interview: four fixed questions in order, at most one follow-up or nudge per question, then two or three sentences of spoken feedback. The session ends on its own after that.

> **TODO before submitting:** fill in every `TODO` in this file and in `PROMPTS.md` with numbers and examples from your own sessions. Delete this note.

## How to run it

Python 3.12 (tested with 3.12.10). Pipecat needs 3.11 or newer.

1. Sign up and create an API key for each provider (all have free tiers):
   - Deepgram (speech-to-text): https://console.deepgram.com
   - Groq (LLM): https://console.groq.com
   - Cartesia (text-to-speech): https://play.cartesia.ai
2. Set up the environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env            # then paste your keys into .env
   ```
3. Run it:
   ```bash
   python bot.py
   ```
4. Open http://localhost:7860, click **Connect**, and allow the microphone. Headphones are a good idea, though the browser's echo cancellation usually keeps the agent from hearing itself.

Each session writes to `sessions/<timestamp>/`: `latency.jsonl`, `transcript.txt` and `latency_summary.txt`. To summarise any latency file again, run `python session_log.py path/to/latency.jsonl`. To score a transcript, run `python eval_transcript.py path/to/transcript.txt`.

**Windows note:** Pipecat has some long file paths. If you get `ModuleNotFoundError` for a module that exists on disk, your project or venv path is over Windows' 260-character limit. Move the venv somewhere short such as `C:\venvs\trainertwin`.

## Architecture

```
browser mic ──WebRTC──▶ transport.input()
                          │ 16 kHz PCM
                          ▼
                     Deepgram STT (nova-3, streaming)       → TranscriptionFrame
                          ▼
                     user aggregator                        Silero VAD + Smart Turn v3
                          │                                 decide when the turn is over,
                          │                                 and trigger interruptions
                          ▼
                     InterviewDirector  (custom)            rewrites the system prompt's
                          │                                 "current step" before each LLM call
                          ▼
                     Groq LLM (gpt-oss-120b, streaming)     → LLMTextFrame
                          ▼
                     Cartesia TTS (sonic, websocket)        → audio frames
                          ▼
                     transport.output() ──WebRTC──▶ browser speaker
                          ▼
                     assistant aggregator                   adds what was actually spoken to context

LatencyObserver (custom) watches frames across the whole pipeline → latency.jsonl
Aggregator turn events → transcript.txt
```

Transport: Pipecat's `SmallWebRTCTransport` with its prebuilt browser UI, run by Pipecat's development runner. A browser gives me echo cancellation and noise suppression for free, so local mic tests work without the agent interrupting itself.

Custom pieces:

- **`interview.py` → `InterviewDirector`** (a `FrameProcessor`). Before every LLM call it reads the conversation so far and works out the state: which of the four questions was last asked (matched on the exact question wording) and how many user replies have come since. It then puts one instruction into the system prompt: ask question 1; or "decide: one follow-up, one nudge, or move on to question N+1"; or "follow-up already used, move on now"; or "give the closing feedback". The LLM never has to track state itself. This also caps follow-ups at one per question, and I avoided a tool call, which would add a second LLM round trip. After the closing line has been spoken, and the user hasn't interrupted, it waits 1.5 s and ends the session.
- **`session_log.py` → `LatencyObserver`** (a pipeline observer) builds one row per user turn. It also writes the summary. See the latency section.

## Providers and why

| Stage | Choice | Why | Trade-off |
|---|---|---|---|
| STT | Deepgram nova-3 (streaming) | Fast finals (typically a few hundred ms), good with Indian English accents and technical vocabulary, generous free credit | Paid after credit; occasional mis-hearing of product names |
| LLM | Groq `openai/gpt-oss-120b`, reasoning effort low | Very low time-to-first-token on Groq hardware, and good enough at following the step instruction. Pipecat 1.14's default Groq model. | Reasoning tokens add a little delay before the first spoken token; free-tier rate limits |
| TTS | Cartesia sonic | Lowest time-to-first-audio I know of over a websocket, natural voice, word timestamps (so interrupted turns store only what was actually spoken) | Smaller voice catalogue than ElevenLabs |

TODO: add anything you saw in practice, for example "Deepgram heard 'Kafka' as 'cafe car'".

## Turn-taking and interruptions

| Setting | Default | Mine | Why |
|---|---|---|---|
| VAD `stop_secs` | 0.2 | 0.2 | Short on purpose. It only decides when to ask Smart Turn "is the turn finished?"; it does not end the turn by itself. |
| Smart Turn v3 (`LocalSmartTurnAnalyzerV3`) | on | on | An audio model that hears whether the speaker sounds finished, from intonation and trailing words like "um". This is what makes the agent wait on "I worked on… um…" (scenario B) but reply quickly after a falling "…payments service." |
| Smart Turn `stop_secs` (fallback) | 3.0 | 3.0 (env `SMART_TURN_STOP_SECS`) | If Smart Turn keeps saying "not finished", the turn ends anyway after this much silence. In an interview people pause to think, so I kept it long rather than cut them off. TODO: say whether you tried lower, e.g. 1.5 s, and what happened. |
| VAD `start_secs` | 0.2 | 0.2 | How long the user must speak before it counts as speech and interrupts the bot. 0.2 s stops the bot quickly in scenario C. TODO: if coughs or "mm-hmm" interrupted it, say so; raising this or adding `MinWordsUserTurnStartStrategy` is the fix. |
| VAD `confidence` | 0.7 | 0.7 | Left at default. |
| Interruptions | on | on | The VAD start interrupts the bot. Pipecat cancels the in-flight LLM and TTS and flushes queued audio. Cartesia's word timestamps mean the context only keeps the words that were actually played, so the LLM knows where it was cut off. The system prompt tells it to respond to what the user said, then carry on from the current step. |

What went wrong before I tuned these: TODO, from your own testing. For example: with VAD-only end-of-turn (no Smart Turn) and `stop_secs=0.8`, "I worked on… um…" got answered mid-thought; with `stop_secs=1.5` every reply felt slow.

## Latency

How each field in `latency.jsonl` is captured (timestamps are ms since the session started):

- `user_stopped_ms`: the VAD stop time minus `stop_secs`, which is when the user actually went quiet, not when detection fired. After a mid-answer pause, the last stop counts.
- `stt_final_ms`: arrival of the last final transcript before the LLM is called.
- `llm_request_ms` (extra field): when the turn was declared over and the context was sent to the LLM. It separates turn-detection wait from LLM time.
- `llm_first_token_ms`: the first text token from the LLM.
- `tts_first_audio_ms`: `BotStartedSpeakingFrame`, when the server starts sending the reply's audio over WebRTC. **Approximation:** browser playout comes after network and jitter-buffer delay, typically 20–80 ms on localhost, which the server can't see. The true number is slightly higher.
- `interrupted`: the user started speaking while this reply was playing.

Results from the recorded session (`latency.jsonl`, N = TODO turns):

| | ms |
|---|---|
| user stopped → first audio, **p50** | TODO |
| user stopped → first audio, **p95** | TODO |
| mean: user stopped → STT final | TODO |
| mean: STT final → turn end (Smart Turn decision) | TODO |
| mean: turn end → LLM first token | TODO |
| mean: LLM first token → first audio (TTS) | TODO |

Paste these from `latency_summary.txt`. Where the time goes: TODO. Usually the biggest piece is end-of-turn detection, the 200 ms VAD wait plus Smart Turn's verdict, and longer when Smart Turn waits after a trailing "um". Next is LLM time-to-first-token, then TTS first audio.

## What's weak and what I'd do next

- Question tracking matches on the exact question wording. If the LLM paraphrases a question, the director thinks it hasn't been asked and asks again. If a follow-up happens to use a later question's words ("how did you know it was working?"), it counts as that question. Next: have the LLM emit a short hidden tag that is stripped before TTS, or use a tool call off the latency path.
- The closing feedback is generated from the conversation in context. It is not checked against a rubric while the session runs.
- `tts_first_audio_ms` is measured on the server; I'd add client-side playout timestamps from the browser via RTVI to close that gap.
- The eval script uses an LLM judge from the same model family. I'd add a few hand-scored transcripts to check it.
- TODO: anything you noticed in your own sessions.

## Stretch

- **Eval:** `eval_transcript.py` scores a transcript for question order, grounded reactions, follow-up judgement, feedback specificity and voice style (1–5 each).
- **Persona:** "Maya" is warm but honest; the persona and rules are in `PROMPTS.md`.

## AI tools

- Claude Code (Claude Opus 5.5): scaffolded the project, read the Pipecat 1.14 source to find the current APIs, and drafted `bot.py`, `interview.py`, `session_log.py`, the eval script and these docs. I reviewed, ran and tuned everything. TODO: adjust to what you actually did.

## Time spent

TODO: roughly N hours.
