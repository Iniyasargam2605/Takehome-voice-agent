"""Interview script, prompts, and the processor that keeps the LLM on track.

The LLM does the talking, but it does not decide where we are in the
interview. ``InterviewDirector`` reads the conversation so far before every
LLM call, works out which question we are on and how many replies the user
has given to it, and rewrites the system prompt's "current step" line. That
keeps the 4 questions in order and caps follow-ups at one per question,
without a tool call (which would add a second LLM round trip per turn).
"""

import asyncio
import re

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndTaskFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

QUESTIONS = [
    "Tell me about a project you're proud of.",
    "What was the hardest technical problem in it?",
    "How did you know it was working well?",
    "What would you do differently now?",
]

# Phrases used to spot which question an agent message asked. The prompt tells
# the LLM to use the exact wording. Each phrase runs to the end of the
# question, so a question cut off by an interruption doesn't count as asked.
QUESTION_SIGNATURES = [
    ("project you're proud of", "project you are proud of"),
    ("hardest technical problem in it",),
    ("know it was working well",),
    ("do differently now",),
]

CLOSING_LINE = "That's the end of our practice session."

SYSTEM_PROMPT = f"""You are Maya, a friendly but honest interview coach at TrainerTwin. You are running a short spoken mock interview to help the candidate practise.

How you speak:
- Everything you write is spoken aloud by a text-to-speech voice. Write plain spoken English only.
- Never use lists, bullet points, markdown, emojis, headings or stage directions.
- Keep every reply to one to three short sentences. One idea per sentence.
- Say numbers the way a person would say them, for example "about two hundred milliseconds", not "~200ms".
- Do not spell out or repeat back names, URLs or acronyms you are unsure of. Refer to "your project" or "that service" instead.
- Sound warm and natural, like a real interviewer. Use the candidate's own words when you react.

The interview has exactly four questions, asked in this order:
1. {QUESTIONS[0]}
2. {QUESTIONS[1]}
3. {QUESTIONS[2]}
4. {QUESTIONS[3]}

Rules:
- When you ask one of these four questions, use its exact wording.
- Before asking the next question, react to what the candidate actually said in a few words. Do not just say "great".
- A follow-up is one short question about something specific they said, for example asking for a number, their own role, or a concrete example.
- A nudge is a gentle prompt when the answer was empty, off topic, or "I don't know", for example "Take your time. Even a small project counts."
- Never ask more than one question in a single reply.
- Never skip a question, reorder them, or invent new main questions.
- If the candidate interrupts you, stop and respond to what they just said, then continue from the current step.
- If the candidate asks you something, answer in one sentence, then continue from the current step.

Current step:
{{step}}"""

STEP_GREETING = (
    "The session is starting. Greet the candidate in one short sentence, say this is a four question "
    f'practice interview, then ask question 1 using its exact wording: "{QUESTIONS[0]}"'
)

STEP_FIRST_REPLY = (
    "You asked question {n}. The candidate has just answered it. Decide:\n"
    "- If the answer was vague, very short, or missing a concrete detail, ask ONE short follow-up about it.\n"
    "- If the answer was empty, off topic, or they said they don't know, give ONE gentle nudge.\n"
    "- If they asked you something instead of answering, answer it in one sentence, then ask question "
    "{n} again.\n"
    "- Otherwise, react briefly to what they said and move on: {next_action}"
)

STEP_SECOND_REPLY = (
    "You asked question {n} and already used your one follow-up or nudge. Do not ask anything more about it. "
    "React briefly to what they said, then {next_action}"
)

# The second sentence matters: a question cut off by an interruption stays in
# the context, and the LLM otherwise assumes it was asked and skips ahead.
NEXT_QUESTION = (
    'ask question {n} using its exact wording: "{q}" If you started asking it earlier and were cut off, '
    "it has not been asked yet, so ask it in full now."
)

FEEDBACK = (
    "give your closing feedback now. In two or three short spoken sentences: name one real strength "
    "from their answers, quoting a specific thing they said, and one concrete thing to improve next time, "
    "such as giving numbers, explaining their own role, or structuring the answer. Then say exactly: "
    f'"{CLOSING_LINE}"'
)

STEP_AFTER_FEEDBACK = (
    "The interview is over and you already gave feedback. Reply in one sentence to what they said, then "
    f'say exactly: "{CLOSING_LINE}"'
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()


def asked_question(text: str) -> int | None:
    """Return the 1-based number of the question this agent message asks, if any."""
    t = _normalise(text)
    asked = None
    for i, signatures in enumerate(QUESTION_SIGNATURES, start=1):
        if any(s in t for s in signatures):
            asked = i
    return asked


def current_step(messages: list[dict]) -> str:
    """Work out the instruction for the next agent reply from the conversation."""
    question = 0
    replies_since_question = 0
    feedback_given = False
    previous_role = None
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if role == "assistant":
            if CLOSING_LINE.lower().rstrip(".") in _normalise(content):
                feedback_given = True
            n = asked_question(content)
            # >= so a re-asked question (after the user asked something
            # instead of answering) gets a fresh follow-up allowance.
            if n and n >= question:
                question = n
                replies_since_question = 0
        elif role == "user" and previous_role != "user":
            # Smart Turn can end a turn at a sentence break and the user then
            # carries on, leaving several user messages in a row. That's still
            # one reply.
            replies_since_question += 1
        if role in ("user", "assistant") and (role == "user" or content.strip()):
            previous_role = role

    if feedback_given:
        return STEP_AFTER_FEEDBACK
    if question == 0:
        return STEP_GREETING

    if question < len(QUESTIONS):
        next_action = NEXT_QUESTION.format(n=question + 1, q=QUESTIONS[question])
    else:
        next_action = FEEDBACK

    if replies_since_question <= 1:
        return STEP_FIRST_REPLY.format(n=question, next_action=next_action)
    return STEP_SECOND_REPLY.format(n=question, next_action=next_action)


def build_system_prompt(step: str) -> str:
    return SYSTEM_PROMPT.format(step=step)


class InterviewDirector(FrameProcessor):
    """Sits between the user aggregator and the LLM.

    - On every ``LLMContextFrame`` it rewrites the system message with the
      current step, so the LLM always knows exactly what it is allowed to do.
    - After the closing line has been spoken (and not interrupted), it ends
      the session.
    """

    def __init__(self, end_delay_secs: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self._context: LLMContext | None = None
        self._end_delay_secs = end_delay_secs
        self._end_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            self._context = frame.context
            self._update_system_prompt(frame.context)
        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame, BotStartedSpeakingFrame)):
            await self._cancel_end()
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._closing_line_spoken():
            # Wait a moment: BotStoppedSpeaking can fire on a short gap between
            # sentences, and the user may still want to say something.
            await self._cancel_end()
            self._end_task = self.create_task(self._end_after_delay())

        await self.push_frame(frame, direction)

    def _update_system_prompt(self, context: LLMContext):
        messages = context.get_messages()
        step = current_step(messages)
        logger.info(f"Interview step: {step.splitlines()[0][:120]}")
        prompt = {"role": "system", "content": build_system_prompt(step)}
        if messages and messages[0].get("role") == "system":
            messages = [prompt, *messages[1:]]
        else:
            messages = [prompt, *messages]
        context.set_messages(messages)

    def _closing_line_spoken(self) -> bool:
        if not self._context:
            return False
        messages = self._context.get_messages()
        if not messages or messages[-1].get("role") != "assistant":
            return False
        content = messages[-1].get("content")
        return isinstance(content, str) and CLOSING_LINE.lower().rstrip(".") in _normalise(content)

    async def _end_after_delay(self):
        await asyncio.sleep(self._end_delay_secs)
        logger.info("Closing line spoken, ending session")
        await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    async def _cancel_end(self):
        if self._end_task:
            await self.cancel_task(self._end_task)
            self._end_task = None
