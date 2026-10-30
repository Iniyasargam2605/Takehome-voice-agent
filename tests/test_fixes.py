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
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import interview as iv
import session_log
import turn_end
from deepgram.listen.v1.types import ListenV1Results
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
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
        self.assertIn("ask ONE short follow-up", iv.current_step(messages))

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
            coach("<answer> Nice. " + Q[2]), user("Technical or business?"),
            coach("<question> Either works. " + Q[2]), user("we tracked lag"),
        ]
        step = iv.current_step(messages)
        self.assertTrue(first_line(step).startswith("You asked question 3."))
        self.assertIn("ask ONE short follow-up", step)

    def test_step_offers_every_tag(self):
        step = iv.current_step([coach(Q[0]), user("a")])
        for tag in iv.TAGS:
            self.assertIn(f"<{tag}>", step)
        self.assertIn("Repeat your last question word for word", step)

    def test_feedback_after_question_4(self):
        messages = [coach(Q[0]), user("a"), coach(Q[1]), user("b"), coach(Q[2]), user("c"), coach(Q[3]), user("d"), coach("Why?"), user("e")]
        self.assertIn("closing feedback", iv.current_step(messages))


class TaggedTurnTests(unittest.TestCase):
    """Bug 7: "You repeat?" counted as the answer to a follow-up, and "You ask
    the question again?" got a nudge instead of the question. The LLM now tags
    what each turn was, so any wording of a repeat request is handled."""

    def test_repeat_request_does_not_use_follow_up(self):
        messages = [coach(Q[0]), user("mumble"), coach("<unclear> Sorry, I didn't catch that. Could you say it again?"),
                    user("I built a payments service")]
        self.assertEqual(iv.interview_state(messages)[:3], (1, 0, 1))
        self.assertIn("ask ONE short follow-up", iv.current_step(messages))

    def test_repeated_follow_up_then_answer_moves_on(self):
        messages = [coach(Q[0]), user("a"), coach("<answer> What was your role?"), user("pardon?"),
                    coach("<repeat> What was your role?"), user("I led it")]
        self.assertEqual(iv.interview_state(messages)[:3], (1, 1, 1))
        step = iv.current_step(messages)
        self.assertIn("already used your one follow-up", step)
        self.assertIn(Q[1], step)

    def test_repeat_names_the_follow_up_not_the_main_question(self):
        # Live run: told to "repeat your last question", the LLM repeated
        # question 2 instead of its follow-up. The step now spells it out.
        messages = [coach(Q[0]), user("a"), coach("<answer> Nice impact. What was your specific role in it?"),
                    user("Sorry, could you say it again?")]
        self.assertEqual(iv.last_question(messages, 1), "What was your specific role in it?")
        self.assertIn('word for word: "What was your specific role in it?"', iv.current_step(messages))

    def test_repeat_limit_moves_on(self):
        messages = [coach(Q[0]), user("what?"), coach("<repeat> " + Q[0]), user("huh"),
                    coach("<unclear> Sorry, could you say that again?"), user("blah")]
        step = iv.current_step(messages)
        self.assertTrue(step.startswith("You asked question 1 and have already repeated"))
        self.assertIn(Q[1], step)

    def test_skip_moves_on(self):
        messages = [coach(Q[0]), user("a"), coach("<skip> That's fine. " + Q[1]), user("b")]
        self.assertEqual(iv.interview_state(messages)[:3], (2, 0, 0))

    def test_untagged_or_unknown_tag_counts_as_answer(self):
        for reply in ("What was your role?", "<answered> What was your role?"):
            messages = [coach(Q[0]), user("a"), coach(reply), user("b")]
            self.assertIn("already used your one follow-up", iv.current_step(messages))

    def test_tag_spellings(self):
        self.assertEqual(iv.reply_tag("<Off-Topic> Let's get back."), "off_topic")
        self.assertEqual(iv.reply_tag("  <repeat>Sure."), "repeat")
        self.assertIsNone(iv.reply_tag("Sure. <repeat>"))


class ReplyTagFilterTests(unittest.TestCase):
    def run_filter(self, chunks, bot_spoke=True, interrupted=False):
        director = iv.InterviewDirector()
        f = iv.ReplyTagFilter(director)
        out = []

        async def push(frame, direction=FrameDirection.DOWNSTREAM):
            out.append(frame)

        f.push_frame = push

        async def go():
            frames = [LLMFullResponseStartFrame(), *[LLMTextFrame(c) for c in chunks], LLMFullResponseEndFrame()]
            for frame in frames:
                await f.process_frame(frame, FrameDirection.DOWNSTREAM)
            if interrupted:
                await f.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            if bot_spoke:
                await f.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

        asyncio.run(go())
        spoken = "".join(x.text for x in out if isinstance(x, LLMTextFrame))
        return spoken, director._spoken_tag

    def test_tag_split_across_tokens_is_stripped(self):
        self.assertEqual(self.run_filter(["<rep", "eat", ">", " Sure,", " here."]), (" Sure, here.", "repeat"))

    def test_untagged_reply_passes_through(self):
        self.assertEqual(self.run_filter(["Hi there", ", welcome."]), ("Hi there, welcome.", None))

    def test_unknown_tag_is_stripped_but_not_recorded(self):
        self.assertEqual(self.run_filter(["<maybe> Okay."]), ("Okay.", None))

    def test_reply_cancelled_before_audio_is_not_recorded(self):
        self.assertEqual(self.run_filter(["<answer> Nice."], bot_spoke=False, interrupted=True)[1], None)

    def test_director_restores_tag_on_spoken_reply(self):
        director = iv.InterviewDirector()
        director.tag_spoken("repeat")
        messages = director._restore_tag([coach(Q[0]), user("again?"), coach(Q[0])])
        self.assertEqual(messages[-1]["content"], "<repeat> " + Q[0])
        self.assertEqual(messages[0]["content"], Q[0])
        # Only once.
        self.assertEqual(director._restore_tag(messages), messages)


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
        real_stt = bot.DeepgramSTT

        class RecordingSTT(real_stt):
            def __init__(self, **kwargs):
                captured["stt"] = kwargs
                super().__init__(**kwargs)

        def fake_pair(context, user_params=None, **kwargs):
            captured["params"] = user_params
            raise _Captured

        with tempfile.TemporaryDirectory() as tmp, patch.object(bot, "LLMContextAggregatorPair", fake_pair), patch.object(
            bot, "DeepgramSTT", RecordingSTT
        ), patch.object(
            bot, "SessionLog", lambda: session_log.SessionLog(Path(tmp))
        ):
            with self.assertRaises(_Captured):
                asyncio.run(bot.run_bot(None, None))

        params = captured["params"]
        start = params.user_turn_strategies.start
        self.assertEqual([type(s).__name__ for s in start], ["VADUserTurnStartStrategy"])
        self.assertEqual([type(s).__name__ for s in params.user_turn_strategies.stop], ["SmartTurnStopStrategy"])
        self.assertEqual(params.user_turn_stop_timeout, bot.SMART_TURN_STOP_SECS)
        self.assertGreaterEqual(params.user_turn_stop_timeout, 3.0)
        # Bug 5: the 0.35 s default STT budget was shorter than Deepgram's
        # measured finalisation time, so answers lost their last sentence.
        self.assertGreaterEqual(captured["stt"]["ttfs_p99_latency"], 0.6)



def deepgram_result(transcript, is_final=True, from_finalize=True):
    return ListenV1Results.model_construct(
        is_final=is_final, from_finalize=from_finalize, channel={"alternatives": [{"transcript": transcript}]}
    )


class FinalizeTests(unittest.TestCase):
    """Bug 6: the 0.8 s STT budget was waited out on every turn, because
    Deepgram's reply to finalize is empty when it already sent the text."""

    def test_empty_finalize_detected(self):
        self.assertTrue(turn_end.is_empty_finalize(deepgram_result("")))
        self.assertFalse(turn_end.is_empty_finalize(deepgram_result("the end.")))
        self.assertFalse(turn_end.is_empty_finalize(deepgram_result("", from_finalize=False)))
        self.assertFalse(turn_end.is_empty_finalize(deepgram_result("", is_final=False)))

    def make_strategy(self, vad_stopped, text):
        strategy = turn_end.SmartTurnStopStrategy(turn_analyzer=MagicMock())
        strategy.trigger_user_turn_stopped = AsyncMock()
        strategy._turn_complete = True
        strategy._vad_stopped = vad_stopped
        strategy._text = text
        return strategy

    def test_confirmation_ends_turn_without_waiting(self):
        strategy = self.make_strategy(vad_stopped=True, text="payments service.")
        asyncio.run(strategy.process_frame(turn_end.TranscriptFinalizedFrame()))
        strategy.trigger_user_turn_stopped.assert_awaited_once()

    def test_confirmation_ignored_while_user_still_talking(self):
        # The user spoke again after the VAD stop: more text is on its way.
        strategy = self.make_strategy(vad_stopped=False, text="payments service.")
        asyncio.run(strategy.process_frame(turn_end.TranscriptFinalizedFrame()))
        strategy.trigger_user_turn_stopped.assert_not_awaited()

    def test_confirmation_ignored_without_text(self):
        strategy = self.make_strategy(vad_stopped=True, text="")
        asyncio.run(strategy.process_frame(turn_end.TranscriptFinalizedFrame()))
        strategy.trigger_user_turn_stopped.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
