# TrainerTwin take-home: voice interview coach

A cascaded voice agent (speech-to-text → LLM → text-to-speech) built with Pipecat. "Maya", a friendly interview coach, runs a short mock interview: four fixed questions in order, at most one follow-up or nudge per question, then two or three sentences of spoken feedback. The session ends on its own after that.

> **TODO before submitting:** fill in the remaining `TODO`s (recorded-session latency, time spent, AI tools) with your own numbers. Delete this note.

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

**Automated test session:** with the bot running, `python tools/simulated_candidate.py` connects over WebRTC like a browser. It acts as a candidate (answers from Groq, voice from Cartesia) and plays out scenarios A, B and C, measuring reply gaps and interruption stop time on the client side.

**Tests:** `python -m unittest discover -s tests -v` runs 27 offline regression tests for the bugs below and the intent-tag state machine (no API keys or network needed). After a simulated session, `python tests/check_session.py` checks the newest `sessions/` folder and `bot.log` against each fix: questions in order, follow-ups, scenarios B and C, no phantom turns, latency rows, and that the coach never replied mid-answer.

**Windows note:** Pipecat has some long file paths. If you get `ModuleNotFoundError` for a module that exists on disk, your project or venv path is over Windows' 260-character limit. Move the venv somewhere short such as `C:\venvs\trainertwin`.

## Architecture

```
browser mic ──WebRTC──▶ transport.input()
                          │ 16 kHz PCM
                          ▼
                     Deepgram STT (nova-3, streaming)       → TranscriptionFrame
                          ▼
                     user aggregator                        Silero VAD starts turns (and interrupts);
                          │                                 Smart Turn v3 decides when they end
                          ▼
                     InterviewDirector  (custom)            rewrites the system prompt's
                          │                                 "current step" before each LLM call
                          ▼
                     Groq LLM (gpt-oss-120b, streaming)     → LLMTextFrame, starting with an intent tag
                          ▼
                     ReplyTagFilter  (custom)               strips the tag, reports it once audio plays
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

- **`interview.py`: the LLM classifies, Python enforces.**
  - **The LLM** starts each reply with a tag saying what the candidate's turn was: `answer`, `repeat`, `question`, `unclear`, `off_topic` or `skip`. This handles any wording of "can you repeat that?", which a keyword list can't. It comes in the same call as the reply, so it adds no round trip.
  - **`ReplyTagFilter`**, between the LLM and TTS, strips the tag so it's never spoken. Once the reply starts playing, it hands the tag to the director.
  - **`InterviewDirector`**, before the LLM, writes that tag back onto the reply in the context. It then derives the state from the agent's messages in plain Python:
    - which of the four questions was last asked, matched on the end of the question's exact wording, so a question cut off by an interruption doesn't count
    - whether that question's one follow-up or nudge has been used (only `answer`/`skip` replies count)
    - how many repeats or clarifications there have been (capped at two, then it moves on)
  - **The step instruction** offers the LLM only the branches allowed now, with the last question spelled out for repeats.
  - **Ending:** after the closing line has been spoken, and the user hasn't interrupted, the director waits 1.5 s and ends the session.
  - **Why not Pipecat Flows** (built into Pipecat 1.14 as `pipecat.flows`): its function calls run the LLM a second time before the agent speaks, which would add a round trip to most replies.
  - **Why not a separate classifier model:** it would add 150–300 ms to every turn.
- **`session_log.py` → `LatencyObserver`** (a pipeline observer) builds one row per user turn and writes the summary. See the latency section.
- **`turn_end.py`**: a Deepgram STT subclass and a Smart Turn stop strategy subclass that end the turn as soon as Deepgram confirms the transcript is complete (item 5 under turn-taking).
- **`tools/simulated_candidate.py`**: a test client, described above.

## Providers and why

| Stage | Choice | Why | Trade-off |
|---|---|---|---|
| STT | Deepgram nova-3 (streaming) | Final transcript about 0.3–0.5 s after the user stops (measured mean 490 ms), handles technical vocabulary well, generous free credit | Paid after the credit runs out |
| LLM | Groq `openai/gpt-oss-120b`, reasoning effort low | Time-to-first-token about 0.4–0.8 s on Groq (measured mean 566 ms), and it follows the step instruction reliably. Pipecat 1.14's default Groq model. | Reasoning tokens add some delay before the first spoken token; free-tier rate limits |
| TTS | Cartesia sonic | First audio about 150 ms after the first LLM token (measured), natural voice, word timestamps (so interrupted turns store only what was actually spoken) | Smaller voice catalogue than ElevenLabs |

**Fallback TTS:** set `TTS_PROVIDER=deepgram` in `.env` to use Deepgram Aura-2 (`aura-2-thalia-en`) with the Deepgram key instead of Cartesia; the simulated candidate follows the same setting. It has no word timestamps, so an interrupted reply stays in context as whole sentences. In simulated sessions it added about 230 ms (first audio ~380 ms after the first token, against ~150 ms for Cartesia); with the item 5 fix, p50 was 1729 ms and p95 1756 ms over 5 turns. A worse problem: when a reply is cancelled before any audio plays (the candidate resumes after a pause), Pipecat still stores the whole unplayed text in the context. In one run that put an unheard "What would you do differently now?" in the history, so the director counted question 4 as asked. Cartesia stores only words actually played, so use it for real sessions.

## Turn-taking and interruptions

| Setting | Pipecat default | Mine | Why |
|---|---|---|---|
| VAD `stop_secs` | 0.2 | 0.2 | Short on purpose. It only decides when to ask Smart Turn "is the turn finished?"; it does not end the turn by itself. |
| Turn end: Smart Turn v3 (`LocalSmartTurnAnalyzerV3`) | on | on | An audio model that hears whether the speaker sounds finished, from intonation and trailing words like "um". It makes the agent wait on "I worked on… um…" (scenario B) but reply quickly after a falling "…payments service." |
| Smart Turn `stop_secs` (fallback) | 3.0 | 3.0 (`SMART_TURN_STOP_SECS`) | If Smart Turn keeps saying "not finished", the turn ends anyway after this much silence. People pause to think in interviews, so I kept it long. |
| `user_turn_stop_timeout` | 5.0 | 3.0 (`USER_TURN_STOP_TIMEOUT`) | An inactivity timer: with no voice or transcript activity for this long, the turn ends regardless of Smart Turn. 5 s meant 5 s of dead air in the failure below. 1.5 s cut people off mid "um…". It now matches the Smart Turn fallback. |
| Turn start | VAD **or** transcript | **VAD only** | See "what went wrong" below. This was the biggest latency fix. |
| STT final-transcript wait (`ttfs_p99_latency`) | 0.35 (Deepgram) | 0.8 (`STT_TTFS_P99_SECS`) | How long after the user stops Pipecat waits for the final transcript before ending the turn with the text it has. Deepgram took longer than 0.35 s here; see items 4 and 5 below. Normally the turn ends well before this. |
| VAD `start_secs` | 0.2 | 0.2 | How long the user must speak before it counts as speech and interrupts the bot. 0.2 s stops the bot quickly in scenario C. |
| VAD `confidence` | 0.7 | 0.7 | Left at default. |
| Interruptions | on | on | Voice detected while the bot speaks interrupts it. Pipecat cancels the LLM and TTS work in progress and drops queued audio. Cartesia's word timestamps mean the context keeps only the words that were actually played. The system prompt tells the LLM to respond to what the user said, then carry on from the current step. |

**What went wrong before I changed these.** I found these with the simulated candidate, over seven sessions:

1. **Phantom interruptions, 5–6 s replies.** Two of five turns took over 6 s (p95 6.35 s). In the log, Smart Turn correctly ended the turn and the bot started replying. Then a late final transcript of the same speech arrived, and Pipecat's default `TranscriptionUserTurnStartStrategy` treated it as the user speaking again. That interrupted the reply and opened an empty user turn, which waited for the 5 s `user_turn_stop_timeout` ("Empty user turn (interrupted), running LLM"). **Fix:** start turns from the VAD only. Real interruptions still come from the VAD. After the fix there were zero empty turns, and p95 fell to 1.48 s.
2. **Over-tight timeout broke scenario B.** As a first fix for item 1, I lowered `user_turn_stop_timeout` to 1.5 s. The coach then cut in during the 2 s "um…" pause ("Take your time…"), because this timer overrides Smart Turn's "incomplete" verdict. **Fix:** set it back to 3 s, matching Smart Turn's fallback.
3. **Sentence breaks end turns.** Smart Turn sometimes declares a turn complete at the end of a sentence even though the speaker continues. When they continue, the reply in progress is cancelled, so this costs nothing audible. But the interview director had counted each fragment as a separate reply, so follow-ups were never allowed. **Fix:** in `interview.py`, user messages in a row count as one reply.
4. **The coach replied before hearing the last sentence.** In every answer, the last sentence showed up in the transcript after the coach's reply, as the start of the next turn. Pipecat budgets 0.35 s (`ttfs_p99_latency`) for Deepgram's final transcript; here it took 0.37–0.49 s on average. When the budget ran out first, the turn ended with only the text received so far. **Fix:** raise the budget to 0.8 s. In the next run, `tests/check_session.py` found no answer split across a coach reply (7 of 7), with replies still about 1.4 s.
5. **Every turn then waited the full 0.8 s.** Pipecat ends a turn early when the STT marks a transcript as finalized. On VAD stop it asks Deepgram to finalize, but Deepgram has usually already sent the whole utterance, so its finalize reply is empty and nothing is ever marked finalized. The turn-end wait went from 27 ms to 374 ms on average. **Fix:** `turn_end.py` passes that empty confirmation on as a marker frame, and a Smart Turn strategy subclass treats it like a finalized transcript (only after the VAD stop, with text in hand). The wait fell to 50 ms; the 0.8 s budget is still the safety net.

## Latency

How each field in `latency.jsonl` is captured (timestamps are ms since the session started):

- `user_stopped_ms`: the VAD stop time minus `stop_secs`, which is when the user actually went quiet, not when detection fired. After a mid-answer pause, the last stop before the reply counts.
- `stt_final_ms`: arrival of the last final transcript before the reply starts playing.
- `llm_request_ms` (extra field): when the turn was declared over and the context was sent to the LLM. It separates turn-detection wait from LLM time.
- `llm_first_token_ms`: the first text token of the LLM request whose reply was actually played.
- `tts_first_audio_ms`: `BotStartedSpeakingFrame`, when the server starts sending the reply's audio over WebRTC. **Approximation:** browser playout comes after network and jitter-buffer delay, which the server can't see. The simulated candidate measured this on the client: its gaps ran about 100 ms higher than the server's numbers.
- `interrupted`: the user started speaking while this reply was playing.

If Smart Turn ends a turn early and the user carries on, the LLM request for the fragment is cancelled. Each field therefore keeps its latest value before the reply starts playing, so a row describes the attempt that was actually heard.

**Simulated-candidate session** (5 turns, before the STT budget change in item 4; local `sessions/` logs are not committed). A full rerun with every fix was cut short when the Cartesia free credit ran out (HTTP 402); the part that ran replied in about 1.4 s.

| | ms |
|---|---|
| user stopped → first audio, **p50** | **1148** |
| user stopped → first audio, **p95** | **1476** |
| mean: user stopped → STT final | 490 |
| mean: STT final → turn end (Smart Turn decision) | 27 |
| mean: turn end → LLM first token | 566 |
| mean: LLM first token → first audio (TTS) | 153 |

Where the time goes: about 0.2 s is the VAD silence wait (part of the 490 ms before the final transcript). Deepgram finalising is the rest of that 490 ms. Smart Turn itself is nearly free, because its verdict is usually ready when the transcript lands. The LLM's first token (~570 ms, including its short reasoning) is the largest single piece. Cartesia adds ~150 ms. On the client side, the candidate heard replies 1.24–1.62 s after it stopped. The coach stopped talking ~0.7 s after being interrupted, measured on the client; the server reacts in ~20 ms, and the rest is the VAD's 0.2 s start wait plus audio already in flight.

**Recorded session with a real voice** (`latency.jsonl` in this repo, N = TODO turns): p50 TODO ms, p95 TODO ms. Paste from that session's `latency_summary.txt`.

## What's weak and what I'd do next

- **Question tracking matches on exact wording.** If the LLM paraphrases a question, the director thinks it hasn't been asked and asks it again. A follow-up that ends with a later question's words ("…how did you know it was working well?") counts as that question. Next: tag the question number as well as the intent.
- **The LLM sometimes leaves out the intent tag.** It happened in 1 reply of 7 in the simulated run. An untagged reply counts as an answer, so the coach moves on one turn early but never loops. A reply interrupted while playing can also lose its tag. The tags and the extra prompt text cost a little latency; p50 in that run was 1.9 s vs 1.73 s before, though Groq's first-token time varies between runs.
- **Smart Turn hears a spoken question as unfinished.** "Could you say it again?" was judged incomplete in both simulated runs, so the reply waited for the 3 s fallback (about 4.2 s in total). This is probably the rising intonation. Next: let a transcript ending in "?" shorten the fallback.
- **Sentence-break turn ends waste LLM calls.** Smart Turn sometimes ends a turn at a sentence break and the request gets cancelled. That's invisible to the user, but it costs tokens and, for a speaker who pauses ~1 s between sentences, risks a reply starting too early. Next: wait for confirmation from a second signal (an LLM completeness check, `FilterIncompleteUserTurnStrategies`) only on long answers.
- **The closing feedback isn't checked against a rubric while the session runs.**
- **The simulated candidate is cleaner than a person.** Its pauses are synthetic, and it interrupts at a fixed 1.5 s into the reply.
- **The eval script's judge comes from the same model family as the coach.** I'd add a few hand-scored transcripts to check it.

## Stretch

- **Eval:** `eval_transcript.py` scores a transcript for question order, grounded reactions, follow-up judgement, feedback specificity and voice style (1–5 each).
- **Simulated candidate:** `tools/simulated_candidate.py` replays scenarios A, B and C end to end over WebRTC. It found every bug listed above, and `tests/test_fixes.py` pins each one down.
- **Persona:** "Maya" is warm but honest; the persona and rules are in `PROMPTS.md`.

## AI tools

- Claude Code (Claude Opus 5.5): scaffolded the project, read the Pipecat 1.14 source to find the current APIs, drafted `bot.py`, `interview.py`, `session_log.py`, the eval script, the simulated candidate and these docs, and ran the simulated test sessions and diagnosed their logs. I reviewed, ran and tuned everything. TODO: adjust to what you actually did.

## Time spent

TODO: roughly N hours.
