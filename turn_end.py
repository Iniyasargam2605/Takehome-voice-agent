"""End the user's turn as soon as Deepgram confirms the transcript is complete.

When the VAD reports the user stopped, Pipecat asks Deepgram to finalize and
ends the turn on the finalized transcript. But Deepgram has usually already
sent the whole utterance as a normal final, so its finalize reply is empty.
Pipecat then never marks a transcript as finalized and waits out the full STT
budget (``ttfs_p99_latency``) on every turn, which cost about 0.35 s per reply
with our 0.8 s budget.

``DeepgramSTT`` turns that empty reply into a ``TranscriptFinalizedFrame``, and
``SmartTurnStopStrategy`` treats it like a finalized transcript. Smart Turn
still has to say the turn is complete; the budget stays as the safety net for
when the confirmation is slow.
"""

from deepgram.listen.v1.types import ListenV1Results

from pipecat.frames.frames import DataFrame, Frame
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)


class TranscriptFinalizedFrame(DataFrame):
    """Deepgram confirmed a finalize with no new text: the last final transcript
    already covers everything the user said.

    A data frame, not a system frame, so it stays behind that transcript.
    """


def is_empty_finalize(message) -> bool:
    if not isinstance(message, ListenV1Results) or not message.is_final:
        return False
    if not getattr(message, "from_finalize", False):
        return False
    alternatives = message.channel.alternatives if message.channel else []
    return not alternatives or not alternatives[0].transcript


class DeepgramSTT(DeepgramSTTService):
    async def _on_message(self, message):
        await super()._on_message(message)
        if is_empty_finalize(message):
            await self.push_frame(TranscriptFinalizedFrame())


class SmartTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        # Only after the VAD stop that requested the finalize, and with text
        # from this turn; if the user spoke again, the flag was reset.
        if isinstance(frame, TranscriptFinalizedFrame) and self._vad_stopped and self._text:
            self._transcript_finalized = True
            await self._maybe_trigger_user_turn_stopped()
        return await super().process_frame(frame)
