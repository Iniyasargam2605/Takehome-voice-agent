"""Regression tests for the bugs found in the simulated-candidate sessions.

Run with:  python -m unittest discover -s tests -v
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import interview as iv
import session_log
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMContextFrame,
    LLMTextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

Q = iv.QUESTIONS


def coach(text):
    return {"role": "assistant", "content": text}


def user(text):
    return {"role": "user", "content": text}


def first_line(step):
    return step.splitlines()[0]


class DirectorTests(unittest.TestCase):
    """Bugs 1 and 2, plus the re-ask rule."""

    def test_split_answer_counts_as_one_reply(self):
        # Bug 1: Smart Turn split one answer into three user messages, and the
        # director said the follow-up was already used.
        messages = [coach("Hi. " + Q[0]), user("So I worked on"), user("a payments service"), user("cut effort 80%")]
        self.assertTrue(first_line(iv.current_step(messages)).startswith("You asked question 1. The candidate has just answered"))

    def test_follow_up_answer_is_second_reply(self):
        messages = [coach(Q[0]), user("payments"), user("service"), coach("What was your role?"), user("lead")]
        self.assertIn("already used your one follow-up", iv.current_step(messages))

    def test_cut_off_question_is_not_counted(self):
        # Bug 2: "What would you do differently" was interrupted before "now",
        # but still counted as question 4, so the coach went to feedback.
        messages = [
            coach(Q[0]), user("a"), coach(Q[1]), user("b"), coach(Q[2]), user("c"),
            coach("Great numbers. What would you do differently"), user("Sorry, quick question?"),
        ]
        step = iv.current_step(messages)
        self.assertIn("You asked question 3", step)
        self.assertIn(Q[3], step)
        self.assertNotIn("closing feedback", step)
        # The LLM saw its own cut-off question and skipped ahead without this.
        self.assertIn("were cut off, it has not been asked yet", step)

    def test_full_question_still_detected(self):
        self.assertEqual(iv.asked_question("Nice. " + Q[3]), 4)
        self.assertEqual(iv.asked_question("Nice. What’s a project you’re proud of?"), 1)
        self.assertIsNone(iv.asked_question("What would you do differently"))

    def test_reasked_question_gets_fresh_follow_up(self):
        # After the candidate asks something instead of answering, the coach
        # re-asks; the real answer should still be allowed a follow-up.
        messages = [
            coach(Q[0]), user("a"), coach(Q[1]), user("b"),
            coach("Nice. " + Q[2]), user("Technical or business?"),
            coach("Either works. " + Q[2]), user("we tracked lag"),
        ]
        self.assertTrue(first_line(iv.current_step(messages)).startswith("You asked question 3. The candidate has just answered"))

    def test_first_reply_step_mentions_reask(self):
        messages = [coach(Q[0]), user("a")]
        self.assertIn("ask question 1 again", iv.current_step(messages))

    def test_feedback_after_question_4(self):
        messages = [coach(Q[0]), user("a"), coach(Q[1]), user("b"), coach(Q[2]), user("c"), coach(Q[3]), user("d"), coach("Why?"), user("e")]
        self.assertIn("closing feedback", iv.current_step(messages))


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class LatencyTests(unittest.TestCase):
    """Bug 4: rows mixed in cancelled attempts and assistant-side frames."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = FakeClock()
        self.patcher = patch("session_log.time.time", self.clock)
        self.patcher.start()
        self.log = session_log.SessionLog(Path(self.tmp.name))
        self.obs = session_log.LatencyObserver(self.log)
        self.context = LLMContext()

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def push(self, at, frame):
        self.clock.now = 1000.0 + at
        asyncio.run(self.obs.on_push_frame(FramePushed(None, None, frame, FrameDirection.DOWNSTREAM, 0)))

    def vad_stop(self, at):
        self.push(at, VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=1000.0 + at))

    def rows(self):
        return [json.loads(l) for l in self.log.latency_path.read_text(encoding="utf-8").splitlines()]

    def test_split_turn_with_cancelled_request(self):
        self.push(5.0, VADUserStartedSpeakingFrame())
        # Smart Turn ends the turn at a sentence break; the LLM starts...
        self.vad_stop(10.2)
        self.push(10.3, TranscriptionFrame("first part.", "u", ""))
        self.push(10.4, LLMContextFrame(self.context))
        self.push(10.9, LLMTextFrame("Nice"))
        # ...but the user carries on, which cancels that request.
        self.push(11.0, VADUserStartedSpeakingFrame())
        self.vad_stop(13.2)
        self.push(13.4, TranscriptionFrame("second part", "u", ""))
        self.push(13.5, LLMContextFrame(self.context))
        # A late final triggers a second request; this is the reply heard.
        self.push(13.6, TranscriptionFrame("and the end.", "u", ""))
        self.push(13.6, LLMContextFrame(self.context))
        self.push(14.1, LLMTextFrame("Great"))
        self.push(14.3, BotStartedSpeakingFrame())
        # The assistant aggregator's context frame after the reply is ignored.
        self.push(16.0, LLMContextFrame(self.context))
        self.push(17.0, BotStoppedSpeakingFrame())
        self.push(20.0, VADUserStartedSpeakingFrame())

        [row] = self.rows()
        self.assertEqual(
            row,
            {"turn": 1, "user_stopped_ms": 13000, "stt_final_ms": 13600, "llm_request_ms": 13600,
             "llm_first_token_ms": 14100, "tts_first_audio_ms": 14300, "interrupted": False},
        )
        order = [row[k] for k in ("user_stopped_ms", "stt_final_ms", "llm_request_ms", "llm_first_token_ms", "tts_first_audio_ms")]
        self.assertEqual(order, sorted(order))

    def test_barge_in_marks_turn_interrupted(self):
        self.push(1.0, VADUserStartedSpeakingFrame())
        self.vad_stop(3.2)
        self.push(3.4, TranscriptionFrame("answer.", "u", ""))
        self.push(3.5, LLMContextFrame(self.context))
        self.push(4.0, LLMTextFrame("So"))
        self.push(4.2, BotStartedSpeakingFrame())
        self.push(5.0, VADUserStartedSpeakingFrame())  # user talks over the bot
        self.assertTrue(self.rows()[0]["interrupted"])

    def test_greeting_is_not_a_turn(self):
        self.push(0.5, LLMContextFrame(self.context))
        self.push(1.0, LLMTextFrame("Hi"))
        self.push(1.2, BotStartedSpeakingFrame())
        self.push(4.0, BotStoppedSpeakingFrame())
        self.push(5.0, VADUserStartedSpeakingFrame())
        self.assertFalse(self.log.latency_path.exists())


class _Captured(Exception):
    pass


class TurnConfigTests(unittest.TestCase):
    """Bug 3: transcripts started phantom user turns; timeout overrode Smart Turn."""

    def test_vad_only_start_and_timeout_matches_smart_turn(self):
        os.environ.update(DEEPGRAM_API_KEY="x", GROQ_API_KEY="x", CARTESIA_API_KEY="x")
        os.environ.pop("USER_TURN_STOP_TIMEOUT", None)
        import importlib

        with patch("dotenv.load_dotenv"):
            import bot

            bot = importlib.reload(bot)
        captured = {}
        real_stt = bot.DeepgramSTTService

        class RecordingSTT(real_stt):
            def __init__(self, **kwargs):
                captured["stt"] = kwargs
                super().__init__(**kwargs)

        def fake_pair(context, user_params=None, **kwargs):
            captured["params"] = user_params
            raise _Captured

        with tempfile.TemporaryDirectory() as tmp, patch.object(bot, "LLMContextAggregatorPair", fake_pair), patch.object(
            bot, "DeepgramSTTService", RecordingSTT
        ), patch.object(
            bot, "SessionLog", lambda: session_log.SessionLog(Path(tmp))
        ):
            with self.assertRaises(_Captured):
                asyncio.run(bot.run_bot(None, None))

        params = captured["params"]
        start = params.user_turn_strategies.start
        self.assertEqual([type(s).__name__ for s in start], ["VADUserTurnStartStrategy"])
        self.assertEqual(params.user_turn_stop_timeout, bot.SMART_TURN_STOP_SECS)
        self.assertGreaterEqual(params.user_turn_stop_timeout, 3.0)
        # Bug 5: the 0.35 s default STT budget was shorter than Deepgram's
        # measured finalisation time, so answers lost their last sentence.
        self.assertGreaterEqual(captured["stt"]["ttfs_p99_latency"], 0.6)


if __name__ == "__main__":
    unittest.main()
