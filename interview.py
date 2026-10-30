"""Interview script, prompts, and the processors that keep the LLM on track.

Division of work:
- The LLM reads each candidate turn and decides what it was (an answer, a
  request to repeat, a question of their own, ...). It starts its reply with
  that decision as a tag such as ``<repeat>``, then speaks normally.
- ``ReplyTagFilter`` (between the LLM and TTS) strips the tag so it is never
  spoken, and hands it to the director once the reply starts playing.
- ``InterviewDirector`` (before the LLM) stores the tag on that reply in the
  context, then derives the interview state from the context in plain Python:
  which question we are on, whether its one follow-up is used, and how many
  repeats or clarifications there have been. It writes only the allowed
  branches into the system prompt's "current step".

Only an answer (or a skip) uses up a question's allowance, so "can you repeat
that?", in any wording, no longer counts as an answer. The decision comes in
the same LLM call as the reply, so this adds no round trip, unlike a tool call
or a second classifier model. Pipecat's own turn-completion markers use the
same trick.
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
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
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

# What the candidate's last turn was, as tagged by the LLM. Only these use up
# a question's one follow-up or nudge; the rest are turns about the question.
ANSWER_TAGS = {"answer", "skip"}
TAGS = ANSWER_TAGS | {"repeat", "question", "unclear", "off_topic"}
# After this many repeats or clarifications on one question, move on anyway,
# so a bad line or a confused candidate can't loop forever.
MAX_REPEATS = 2
# Any leading tag, known or not, so an invented tag is never spoken either.
TAG_RE = re.compile(r"^\s*<([a-z_\- ]{1,20})>\s*", re.IGNORECASE)

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
- After the greeting, every reply starts with exactly one tag in angle brackets, as the current step describes, for example "<answer> Nice work. What was the hardest technical problem in it?". The tag is never spoken.

Current step:
{{step}}"""

STEP_GREETING = (
    "The session is starting. Greet the candidate in one short sentence, say this is a four question "
    f'practice interview, then ask question 1 using its exact wording: "{QUESTIONS[0]}"'
)

STEP_QUESTION = (
    "You asked question {n}. First decide what the candidate's last message was. Start your reply with "
    "exactly one of these tags (it is removed before speaking), then do what it says:\n"
    "<answer> They answered, even partly or vaguely, or said they don't know. {answer_action}\n"
    "<repeat> They asked you to repeat or say that again, or didn't hear you. Repeat your last question "
    'word for word: "{last_q}"\n'
    "<question> They asked you something else. Answer it in one sentence, then repeat your last question "
    'word for word: "{last_q}"\n'
    "<unclear> Their words make no sense here, probably a speech recognition error. Say you didn't quite "
    "catch that and ask them to say it again.\n"
    "<off_topic> It has nothing to do with the interview. Steer back in one sentence, then repeat your "
    'last question: "{last_q}"\n'
    "<skip> They ask to skip this question. Say that's fine, then {next_action}\n"
    "Your reply must start with the tag."
)

ANSWER_FIRST = (
    "If the answer was vague, very short, or missing a concrete detail, ask ONE short follow-up about it. "
    "If it was empty or they don't know, give ONE gentle nudge. Otherwise react briefly to what they said, "
    "then {next_action}"
)

ANSWER_SECOND = (
    "You already used your one follow-up or nudge, so do not ask anything more about it. React briefly to "
    "what they said, then {next_action}"
)

STEP_REPEAT_LIMIT = (
    "You asked question {n} and have already repeated or clarified several times. Start your reply with "
    "<answer>. Whatever they said, react in a few words, then {next_action}"
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


def reply_tag(text: str) -> str | None:
    """The tag at the start of an agent message, if it is one we know."""
    m = TAG_RE.match(text)
    return _known_tag(m.group(1)) if m else None


def _known_tag(raw: str) -> str | None:
    tag = re.sub(r"[\s-]+", "_", raw.strip().lower())
    return tag if tag in TAGS else None


def asked_question(text: str) -> int | None:
    """Return the 1-based number of the question this agent message asks, if any."""
    t = _normalise(text)
    asked = None
    for i, signatures in enumerate(QUESTION_SIGNATURES, start=1):
        if any(s in t for s in signatures):
            asked = i
    return asked


def interview_state(messages: list[dict]) -> tuple[int, int, int, bool]:
    """Derive (question, answers, repeats, feedback_given) from the agent messages.

    question: the last question fully asked (1-4), or 0 before the greeting.
    answers: agent replies since then to an answer or skip, i.e. follow-ups or
        nudges used. An untagged reply counts as one, as before tags existed.
    repeats: agent replies since then to anything else (repeat requests, the
        candidate's own questions, unclear speech, off topic).

    Only agent messages are read, so a user turn split into several messages
    (Smart Turn ending a turn at a sentence break) can't skew the counts.
    """
    question = answers = repeats = 0
    feedback_given = False
    for m in messages:
        content = m.get("content")
        if m.get("role") != "assistant" or not isinstance(content, str) or not content.strip():
            continue
        if CLOSING_LINE.lower().rstrip(".") in _normalise(content):
            feedback_given = True
        n = asked_question(content)
        if n and n > question:
            question, answers, repeats = n, 0, 0
            continue
        if question:
            if (reply_tag(content) or "answer") in ANSWER_TAGS:
                answers += 1
            else:
                repeats += 1
    return question, answers, repeats, feedback_given


def current_step(messages: list[dict]) -> str:
    """Work out the instruction for the next agent reply from the conversation."""
    question, answers, repeats, feedback_given = interview_state(messages)

    if feedback_given:
        return STEP_AFTER_FEEDBACK
    if question == 0:
        return STEP_GREETING

    if question < len(QUESTIONS):
        next_action = NEXT_QUESTION.format(n=question + 1, q=QUESTIONS[question])
    else:
        next_action = FEEDBACK

    if repeats >= MAX_REPEATS:
        return STEP_REPEAT_LIMIT.format(n=question, next_action=next_action)
    answer_action = (ANSWER_FIRST if answers == 0 else ANSWER_SECOND).format(next_action=next_action)
    return STEP_QUESTION.format(
        n=question, answer_action=answer_action, next_action=next_action, last_q=last_question(messages, question)
    )


def last_question(messages: list[dict], question: int) -> str:
    """The last question the agent asked, which may be a follow-up rather than
    one of the four. Spelled out so "repeat your last question" can't be
    mistaken for the main question."""
    for m in reversed(messages):
        content = m.get("content")
        if m.get("role") == "assistant" and isinstance(content, str) and content.strip():
            text = TAG_RE.sub("", content)
            asked = [q.strip() for q in re.findall(r"[^.!?]*\?", text) if q.strip()]
            if asked:
                return asked[-1]
            break
    return QUESTIONS[question - 1]


def build_system_prompt(step: str) -> str:
    return SYSTEM_PROMPT.format(step=step)


class InterviewDirector(FrameProcessor):
    """Sits between the user aggregator and the LLM.

    - On every ``LLMContextFrame`` it puts the tag of the last spoken reply
      back onto that reply in the context, then rewrites the system message
      with the current step, so the LLM always knows what it is allowed to do.
    - After the closing line has been spoken (and not interrupted), it ends
      the session.
    """

    def __init__(self, end_delay_secs: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self._context: LLMContext | None = None
        self._end_delay_secs = end_delay_secs
        self._end_task: asyncio.Task | None = None
        self._spoken_tag: str | None = None

    def tag_spoken(self, tag: str):
        """Called by ``ReplyTagFilter`` when a tagged reply starts playing."""
        self._spoken_tag = tag

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
        messages = self._restore_tag(context.get_messages())
        step = current_step(messages)
        logger.info(f"Interview step: {step.splitlines()[0][:120]}")
        prompt = {"role": "system", "content": build_system_prompt(step)}
        if messages and messages[0].get("role") == "system":
            messages = [prompt, *messages[1:]]
        else:
            messages = [prompt, *messages]
        context.set_messages(messages)

    def _restore_tag(self, messages: list[dict]) -> list[dict]:
        """Prefix the last agent message with the tag its reply was given.

        The context only holds what was spoken, and the tag never is. Putting
        it back keeps the state derivable from the context alone, and shows
        the LLM its own earlier tags.
        """
        tag, self._spoken_tag = self._spoken_tag, None
        if not tag:
            return messages
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content")
                if isinstance(content, str) and content.strip() and not TAG_RE.match(content):
                    messages = list(messages)
                    messages[i] = {**messages[i], "content": f"<{tag}> {content}"}
                break
        return messages

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


class ReplyTagFilter(FrameProcessor):
    """Sits between the LLM and TTS and removes the tag from each reply.

    Holds back the start of each response only until the tag is complete
    (a few tokens), so streaming to TTS is unaffected. Hands the tag to the
    director when the reply starts playing; a reply cancelled before any audio
    played never happened as far as the interview state is concerned.
    """

    def __init__(self, director: InterviewDirector, **kwargs):
        super().__init__(**kwargs)
        self._director = director
        self._buffer = ""
        self._scanning = False
        self._tag: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer, self._scanning, self._tag = "", True, None
        elif isinstance(frame, LLMTextFrame) and self._scanning:
            await self._scan(frame.text)
            return
        elif isinstance(frame, LLMFullResponseEndFrame) and self._scanning:
            await self._flush()
        elif isinstance(frame, BotStartedSpeakingFrame) and self._tag:
            self._director.tag_spoken(self._tag)
            self._tag = None
        elif isinstance(frame, InterruptionFrame):
            self._buffer, self._scanning, self._tag = "", False, None

        await self.push_frame(frame, direction)

    async def _scan(self, text: str):
        self._buffer += text
        head = self._buffer.lstrip()
        if not head:
            return
        m = TAG_RE.match(head)
        if m:
            self._tag = _known_tag(m.group(1))
            logger.info(f"Reply tag: {m.group(1)}")
            self._buffer = head[m.end():]
            await self._flush()
        elif not head.startswith("<") or ">" in head or len(head) > 24:
            # No tag after all (the greeting, or the LLM forgot it).
            await self._flush()

    async def _flush(self):
        self._scanning = False
        if self._buffer:
            await self.push_frame(LLMTextFrame(self._buffer))
            self._buffer = ""
