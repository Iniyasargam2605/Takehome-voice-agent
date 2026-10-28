"""Per-turn latency (latency.jsonl) and transcript logging for one session.

All timestamps are wall-clock ``time.time()`` converted to milliseconds since
the session started (when the bot pipeline was created).

How each latency field is captured:

- ``user_stopped_ms``: the VAD reports a stop only after ``stop_secs`` of
  silence, and stamps the frame with the time it decided. We subtract
  ``stop_secs`` to get when the user actually went quiet. If the user pauses
  and carries on ("I worked on... um... payments"), the later stop wins.
- ``stt_final_ms``: the last final ``TranscriptionFrame`` before the LLM starts.
- ``llm_request_ms`` (extra field): when the turn was declared over and the
  context was sent to the LLM. Splits turn-detection wait from LLM latency.
- ``llm_first_token_ms``: the first ``LLMTextFrame`` of the reply.
- ``tts_first_audio_ms``: ``BotStartedSpeakingFrame``, which the output
  transport emits when it starts writing the reply's audio to the WebRTC
  track. Browser playout adds network and jitter-buffer delay on top (tens
  of milliseconds on localhost) that the server can't observe.
- ``interrupted``: the user started speaking while this reply was playing.
"""

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMContextFrame,
    LLMTextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


@dataclass
class Turn:
    turn: int
    user_stopped_ms: int | None = None
    stt_final_ms: int | None = None
    llm_request_ms: int | None = None
    llm_first_token_ms: int | None = None
    tts_first_audio_ms: int | None = None
    interrupted: bool = False


class SessionLog:
    """Owns the session clock and the output files."""

    def __init__(self, root: Path = Path("sessions")):
        self.start = time.time()
        self.dir = root / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.latency_path = self.dir / "latency.jsonl"
        self.transcript_path = self.dir / "transcript.txt"
        self.turns: list[Turn] = []
        logger.info(f"Logging session to {self.dir}")

    def ms(self, t: float | None = None) -> int:
        return round(((t if t is not None else time.time()) - self.start) * 1000)

    def write_turn(self, turn: Turn):
        self.turns.append(turn)
        with self.latency_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(turn)) + "\n")

    def write_transcript(self, speaker: str, text: str, interrupted: bool = False, at: float | None = None):
        mark = " [interrupted by user]" if interrupted else ""
        line = f"[{self.ms(at) / 1000:7.2f}s] {speaker}: {text.strip()}{mark}"
        logger.info(line)
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_summary(self):
        summary = summarise([asdict(t) for t in self.turns])
        if summary:
            (self.dir / "latency_summary.txt").write_text(summary, encoding="utf-8")
            logger.info("\n" + summary)


class LatencyObserver(BaseObserver):
    """Builds one ``Turn`` per user turn → agent reply and writes it out."""

    def __init__(self, log: SessionLog, **kwargs):
        super().__init__(**kwargs)
        self._log = log
        self._turn: Turn | None = None
        self._bot_speaking = False
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        # A frame is observed once per hop (and some go both directions), so
        # only handle the first time we see each one.
        if frame.id in self._seen:
            return
        if not isinstance(
            frame,
            (
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                TranscriptionFrame,
                LLMContextFrame,
                LLMTextFrame,
                BotStartedSpeakingFrame,
                BotStoppedSpeakingFrame,
            ),
        ):
            return
        self._seen.add(frame.id)
        now = time.time()

        if isinstance(frame, VADUserStartedSpeakingFrame):
            if self._turn and self._turn.tts_first_audio_ms is not None:
                # The previous reply has started playing, so this is a new
                # user turn. If it is still playing, the user barged in.
                self._turn.interrupted = self._bot_speaking
                self._flush()
            if self._turn is None:
                self._turn = Turn(turn=len(self._log.turns) + 1)
            else:
                # Resumed after a pause before we answered: any LLM work for
                # the earlier fragment gets cancelled, so measure afresh.
                self._turn.llm_request_ms = None
                self._turn.llm_first_token_ms = None
            return

        # Until the reply starts playing, each field keeps its latest value,
        # so it describes the attempt that was actually heard. Smart Turn can
        # end a turn at a sentence break, and a late transcript can trigger a
        # second LLM request, so earlier values may belong to cancelled work.
        answering = self._turn is not None and self._turn.tts_first_audio_ms is None
        if isinstance(frame, VADUserStoppedSpeakingFrame) and answering:
            self._turn.user_stopped_ms = self._log.ms(frame.timestamp - frame.stop_secs)
        elif isinstance(frame, TranscriptionFrame) and answering:
            self._turn.stt_final_ms = self._log.ms(now)
        elif isinstance(frame, LLMContextFrame) and answering:
            self._turn.llm_request_ms = self._log.ms(now)
            self._turn.llm_first_token_ms = None
        elif (
            isinstance(frame, LLMTextFrame)
            and answering
            and self._turn.llm_request_ms is not None
            and self._turn.llm_first_token_ms is None
        ):
            self._turn.llm_first_token_ms = self._log.ms(now)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if answering and self._turn.llm_first_token_ms is not None:
                self._turn.tts_first_audio_ms = self._log.ms(now)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

    def _flush(self):
        if self._turn and self._turn.user_stopped_ms is not None:
            self._log.write_turn(self._turn)
        self._turn = None

    def finish(self):
        """Write the last turn and the summary at session end."""
        if self._turn and self._turn.tts_first_audio_ms is not None:
            self._flush()
        self._log.write_summary()


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[round(p) - 1]


def summarise(turns: list[dict]) -> str:
    """p50/p95 of user-stopped → first audio, plus mean time per stage."""
    complete = [
        t
        for t in turns
        if None
        not in (t["user_stopped_ms"], t["stt_final_ms"], t["llm_first_token_ms"], t["tts_first_audio_ms"])
    ]
    if not complete:
        return ""
    total = [t["tts_first_audio_ms"] - t["user_stopped_ms"] for t in complete]
    stt = [t["stt_final_ms"] - t["user_stopped_ms"] for t in complete]
    llm = [t["llm_first_token_ms"] - t["stt_final_ms"] for t in complete]
    # Only present in logs from this app; splits the STT -> LLM stage.
    split = [t for t in complete if t.get("llm_request_ms") is not None]
    turn_wait = [t["llm_request_ms"] - t["stt_final_ms"] for t in split]
    ttft = [t["llm_first_token_ms"] - t["llm_request_ms"] for t in split]
    tts = [t["tts_first_audio_ms"] - t["llm_first_token_ms"] for t in complete]
    lines = [
        f"turns measured: {len(complete)} of {len(turns)}",
        f"user stopped -> first audio: p50 {_percentile(total, 50):.0f} ms, p95 {_percentile(total, 95):.0f} ms",
        "mean per stage:",
        f"  user stopped -> STT final       {statistics.mean(stt):6.0f} ms",
        f"  STT final    -> LLM first token {statistics.mean(llm):6.0f} ms",
    ]
    if split:
        lines += [
            f"    of which: STT final -> turn end / LLM request {statistics.mean(turn_wait):6.0f} ms",
            f"              LLM request -> first token          {statistics.mean(ttft):6.0f} ms",
        ]
    lines += [
        f"  LLM token    -> first audio     {statistics.mean(tts):6.0f} ms",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1] if len(sys.argv) > 1 else "latency.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(summarise(rows) or "no complete turns")
