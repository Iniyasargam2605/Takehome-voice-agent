"""TrainerTwin mock interview coach: a cascaded Pipecat voice agent.

Pipeline:
    browser mic -> SmallWebRTC in -> Deepgram STT -> user aggregator (Silero VAD + Smart Turn v3)
    -> InterviewDirector -> Groq LLM -> Cartesia TTS -> SmallWebRTC out -> browser speaker
    -> assistant aggregator

Run with:  python bot.py   then open http://localhost:7860
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from interview import InterviewDirector
from session_log import LatencyObserver, SessionLog
from turn_end import DeepgramSTT, SmartTurnStopStrategy

load_dotenv(override=True)

# Turn-taking settings. See README "Turn-taking and interruptions".
# VAD stop: how much silence before the VAD says "speech stopped" and asks
# Smart Turn whether the turn is complete. Short, because Smart Turn decides.
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.2"))
# VAD start: how much speech before we treat it as the user talking (and
# interrupt the bot). Lower barges in faster but coughs interrupt too.
VAD_START_SECS = float(os.getenv("VAD_START_SECS", "0.2"))
VAD_CONFIDENCE = float(os.getenv("VAD_CONFIDENCE", "0.7"))
# Smart Turn fallback: if the model keeps saying "not finished", end the turn
# anyway after this much silence. Interview answers have thinking pauses.
SMART_TURN_STOP_SECS = float(os.getenv("SMART_TURN_STOP_SECS", "3.0"))
# Inactivity safety net: a user turn with no voice or transcript activity for
# this long ends regardless of Smart Turn. It overrides Smart Turn's "not
# finished" verdict too (1.5 s cut people off mid "um..."), so keep it equal
# to the Smart Turn fallback. Pipecat's default is 5 s.
USER_TURN_STOP_TIMEOUT = float(os.getenv("USER_TURN_STOP_TIMEOUT", str(SMART_TURN_STOP_SECS)))
# How long after the user stops we wait for the STT's final transcript before
# ending the turn with whatever text we have. Pipecat assumes 0.35 s for
# Deepgram; we measured 0.37-0.49 s on average, so the timer often fired
# first and the last sentence of an answer landed in the next turn. Turns
# normally end sooner than this: see turn_end.py.
STT_TTFS_P99_SECS = float(os.getenv("STT_TTFS_P99_SECS", "0.8"))


def make_tts():
    """Cartesia by default. TTS_PROVIDER=deepgram uses Deepgram Aura-2 with the
    STT key instead; it has no word timestamps, so an interrupted reply is kept
    in context as whole sentences rather than only the words played."""
    if os.getenv("TTS_PROVIDER", "cartesia").lower() == "deepgram":
        return DeepgramTTSService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramTTSService.Settings(
                voice=os.getenv("DEEPGRAM_TTS_VOICE", "aura-2-thalia-en"),
            ),
        )
    return CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
        ),
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    log = SessionLog()

    stt = DeepgramSTT(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        ttfs_p99_latency=STT_TTFS_P99_SECS,
        settings=DeepgramSTT.Settings(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3-general"),
            smart_format=True,
        ),
    )

    reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", "low") or None
    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            reasoning_effort=reasoning_effort,
            temperature=0.6,
            max_completion_tokens=400,
        ),
    )

    tts = make_tts()

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=VAD_CONFIDENCE,
                    start_secs=VAD_START_SECS,
                    stop_secs=VAD_STOP_SECS,
                )
            ),
            user_turn_strategies=UserTurnStrategies(
                # VAD only. The default also starts a turn on any transcript,
                # so a late transcript of speech that had already ended the
                # turn would interrupt the bot's reply and open an empty turn.
                start=[VADUserTurnStartStrategy()],
                stop=[
                    SmartTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams(stop_secs=SMART_TURN_STOP_SECS)
                        )
                    )
                ]
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            InterviewDirector(),
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    latency = LatencyObserver(log)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[latency],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        if message.content:
            log.write_transcript("User", message.content)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        if message.content or message.interrupted:
            log.write_transcript("Coach", message.content or "(cut off before speaking)", message.interrupted)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected, starting interview")
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        latency.finish()


async def bot(runner_args: RunnerArguments):
    """Entry point called by the Pipecat development runner."""
    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import sys

    from pipecat.runner.run import main

    # The runner prints a box-drawing banner; Windows consoles default to cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
