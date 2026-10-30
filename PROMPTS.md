# Prompts

All prompts live in `interview.py`. There is one system prompt with a `{step}` slot. Before every LLM call, `InterviewDirector` fills the slot with exactly one of the step instructions below. There is no separate feedback call: the feedback is the reply generated when the step is "give your closing feedback".

## System prompt (exactly as used)

```text
You are Maya, a friendly but honest interview coach at TrainerTwin. You are running a short spoken mock interview to help the candidate practise.

How you speak:
- Everything you write is spoken aloud by a text-to-speech voice. Write plain spoken English only.
- Never use lists, bullet points, markdown, emojis, headings or stage directions.
- Keep every reply to one to three short sentences. One idea per sentence.
- Say numbers the way a person would say them, for example "about two hundred milliseconds", not "~200ms".
- Do not spell out or repeat back names, URLs or acronyms you are unsure of. Refer to "your project" or "that service" instead.
- Sound warm and natural, like a real interviewer. Use the candidate's own words when you react.

The interview has exactly four questions, asked in this order:
1. Tell me about a project you're proud of.
2. What was the hardest technical problem in it?
3. How did you know it was working well?
4. What would you do differently now?

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
{step}
```

## Step instructions (one goes into `{step}`)

**Greeting** (no question asked yet):
```text
The session is starting. Greet the candidate in one short sentence, say this is a four question practice interview, then ask question 1 using its exact wording: "Tell me about a project you're proud of."
```

**On question N** (the LLM tags what the candidate's turn was, then follows that branch):
```text
You asked question {n}. First decide what the candidate's last message was. Start your reply with exactly one of these tags (it is removed before speaking), then do what it says:
<answer> They answered, even partly or vaguely, or said they don't know. {answer_action}
<repeat> They asked you to repeat or say that again, or didn't hear you. Repeat your last question word for word: "{last_q}"
<question> They asked you something else. Answer it in one sentence, then repeat your last question word for word: "{last_q}"
<unclear> Their words make no sense here, probably a speech recognition error. Say you didn't quite catch that and ask them to say it again.
<off_topic> It has nothing to do with the interview. Steer back in one sentence, then repeat your last question: "{last_q}"
<skip> They ask to skip this question. Say that's fine, then {next_action}
Your reply must start with the tag.
```

`{last_q}` is the last question the agent actually asked (it may be a follow-up), pulled from its last message by code.

`{answer_action}` while the question's follow-up or nudge is unused:
```text
If the answer was vague, very short, or missing a concrete detail, ask ONE short follow-up about it. If it was empty or they don't know, give ONE gentle nudge. Otherwise react briefly to what they said, then {next_action}
```
and once it is used:
```text
You already used your one follow-up or nudge, so do not ask anything more about it. React briefly to what they said, then {next_action}
```

**Repeat limit** (two repeats or clarifications on this question already):
```text
You asked question {n} and have already repeated or clarified several times. Start your reply with <answer>. Whatever they said, react in a few words, then {next_action}
```

`{next_action}` is either the next question:
```text
ask question {n} using its exact wording: "{q}" If you started asking it earlier and were cut off, it has not been asked yet, so ask it in full now.
```
or, after question 4, the feedback prompt:
```text
give your closing feedback now. In two or three short spoken sentences: name one real strength from their answers, quoting a specific thing they said, and one concrete thing to improve next time, such as giving numbers, explaining their own role, or structuring the answer. Then say exactly: "That's the end of our practice session."
```

**After feedback** (the user says something after the closing):
```text
The interview is over and you already gave feedback. Reply in one sentence to what they said, then say exactly: "That's the end of our practice session."
```

## How the agent decides: follow-up, nudge, or next question

The decision is split between code and the LLM:

- **The LLM decides what the candidate's turn was.** It starts each reply with a tag: `answer`, `repeat`, `question`, `unclear`, `off_topic` or `skip`. This works for any wording ("say that again", "sorry, what?", "pardon?"), which a keyword list can't. The decision comes in the same call as the reply, so there is no extra round trip. Pipecat's own turn-completion markers use the same trick.
- **`ReplyTagFilter` strips the tag before TTS** and, once the reply starts playing, hands the tag to the director. The director writes it back onto that reply in the context. A reply cancelled before any audio played gets no tag.
- **Code decides where we are and what is allowed.** `interview_state()` reads only the agent's messages:
  - The last agent message containing the end of a question's exact wording ("…do differently now") sets the current question. A question cut off by an interruption doesn't count.
  - Agent replies since then tagged `answer` or `skip` count as the follow-up or nudge used. An untagged reply counts as one, which was the behaviour before tags.
  - Other tags count as repeats or clarifications. They never use up the follow-up, and after two the step forces a move on, so nothing can loop forever.

  So order is fixed, follow-ups are capped at one per question, and repeats are capped at two, whatever the LLM "wants". The step only offers the branches that are allowed now.
- **The LLM also makes the judgement call** on an answer: a follow-up (vague or missing detail), a nudge (empty or "I don't know"), or no further question.
- The closing line "That's the end of our practice session." is a fixed phrase, so code can tell that feedback was given. After it is spoken without an interruption, the director ends the session.

## How the prompt is shaped for voice

- Plain spoken English only. No lists, markdown or emojis: TTS would read them out or stumble.
- One to three short sentences per reply, one question per reply. Short replies also cut time to first audio, because TTS can start on the first sentence.
- Numbers said the way people say them ("about two hundred milliseconds"), not symbols.
- No spelling out names, URLs or acronyms the STT may have mangled. The agent says "your project" instead of repeating back a possibly wrong product name.
- Reactions must reuse the candidate's own words, so they sound like listening rather than "Great!".
- Only the main questions are asked verbatim. Follow-ups are free-form so they sound natural.

## How the closing feedback is generated

When the director sees that question 4 has been answered (and its follow-up used, or skipped by the LLM), the step becomes the feedback prompt. The LLM sees the whole conversation in context, including only the words that were actually spoken in interrupted turns. It is asked for one real strength quoting something the candidate said, plus one concrete improvement (numbers, own role, structure). Asking for a quote keeps the feedback tied to this session rather than generic.

## Iterations

From the start, the design was code-tracked state plus a per-reply step instruction, rather than one static prompt where the LLM tracks progress itself. The iterations below come from sessions with the simulated candidate (`tools/simulated_candidate.py`). The excerpts are copied from `transcript.txt` and the bot log.

**v1: count every user message as a reply; match short question phrases.**
What went wrong: Smart Turn split one long answer into several user messages, so after the first answer the director already said the follow-up was used up. A cut-off question 4 was counted as asked, the interjection counted as its answer, and the coach went straight to feedback. Question 4 was never answered:
```text
[ 41.85s] User: So I worked on sure. At my last FinTech startup, I built a payments reconciliation service ...
[ 54.01s] User: I designed a flexible matching engine that handles partial matches, ...
  (step: "You asked question 1 and already used your one follow-up or nudge ...")
[138.06s] Coach: Great numbers and reliable alerts. What would you do differently [interrupted by user]
[142.68s] User: Sorry.
[142.94s] User: Quick question. Should I focus on the technical side or the business side?
[165.18s] Coach: Both are valuable, but leading with the technical impact ... That's the end of our practice session.
```

**v2: merge consecutive user messages into one reply; match on the end of each question.**
Result: real follow-ups appeared ("How many transactions did the service handle each day?"), and a cut-off question no longer counted. New problem: when the candidate replied to a fully asked question with a question of their own, the LLM answered it and moved on, so question 3 was skipped:
```text
[115.16s] Coach: Nice speed. How did you know it was working well?
[117.71s] User: Sorry.
[120.67s] User: Quick question.
[120.94s] User: Should I focus on the technical side or the business side?
[127.54s] Coach: Both are valuable, but highlighting the business impact shows why the technical work mattered. What would you do differently now?
```

**v3: tell the LLM to answer a question from the candidate in one sentence and then re-ask; a re-asked question resets the follow-up count.**
Result: the coach answers and returns to the question it was on:
```text
[133.72s] Coach: Nice, a solid 96 percent. What would you do differently [interrupted by user]
[138.37s] User: ... Sorry. Quick question.
[142.40s] Coach: Sure, what's your quick question? What would you do differently now?
```

**v4: say outright that a cut-off question has not been asked.**
What went wrong in v3: the director correctly said "ask question 3", but the LLM saw its own half-asked question 3 in the history and moved on to question 4, so question 3 was never answered:
```text
[115.92s] Coach: Got it, latency was a key concern. How did you know it was working [interrupted by user]
[120.20s] User: settlement file end to end in under five hundred milliseconds ...
[124.62s] Coach: Impressive numbers. What would you do differently now?
```
Fix: the next-question instruction now ends "If you started asking it earlier and were cut off, it has not been asked yet, so ask it in full now." In the next run, the coach was cut off during a follow-up, answered the candidate's interjection, then asked question 3 in full:
```text
[112.46s] Coach: Got it. How many bank file formats did you end up supporting [interrupted by user]
[117.39s] User: Sorry. Quick question. I focus on the technical side or the business side?
[124.29s] Coach: You can highlight both, but lead with the technical challenges since they showcase your problem-solving. How did you know it was working well?
```

**v5: the LLM tags what each turn was; code counts only answers.**
What went wrong in v4 (a manual session): asking for a repeat was counted as an answer, and a second request got a nudge instead of the question:
```text
[103.98s] Coach: What metric or test result showed it passed those evaluation checks?
[108.55s] User: You repeat?
[112.88s] Coach: Nice confirmation. What would you do differently now?
[117.30s] User: You ask the question again?
[123.21s] Coach: Take your time. Even a small change counts—what's one thing you'd tweak if you could start over?
```
A keyword list ("repeat", "again", "pardon") can't cover every wording, so the LLM now classifies the turn with a tag in the same call. The first version only said "repeat your last question". Replayed against Groq, the model left out the tag 2 times in 3, and when it did tag `<repeat>` it repeated question 2 instead of its own follow-up. Two changes fixed that in 6 of 6 replays: the tag rule went into the main rules with an example, and code spells out the last question (`{last_q}`). In the next simulated session:
```text
[ 70.01s] Coach: Impressive impact. How many transactions did the service typically process each day?
[ 80.97s] User: (Sorry, I missed that.) Could you say it again?
[ 85.85s] Coach: How many transactions did the service typically process each day?
[100.89s] User: We are typically pulling in around 120 k settlement records per day, ...
[116.35s] Coach: That's a solid scale. What was the hardest technical problem in it?
```
The repeat request didn't use up the follow-up. After the real answer, the coach moved on because the follow-up had been used. 6 of 7 replies carried a tag. The LLM also invented `<greeting>` once; the filter strips unknown tags without recording them.

## Known failure cases

1. **Paraphrased question → asked twice.** Detection relies on the exact question wording. If the LLM rephrases ("What's a project you're really proud of?"), the director thinks question 1 was never asked and tells the LLM to ask it again.
2. **Follow-up that sounds like a later question.** A follow-up on question 2 ending "…how did you know it was working well?" matches question 3's wording, so the director skips ahead and question 3 is never asked as such.
3. **A question split by a sentence-break turn end.** In v3's run, Smart Turn ended the turn after "Sorry. Quick question.", before the actual question. The coach replied "Sure, what's your quick question?" and asked question 4 in the same breath. The candidate's real question then arrived merged with their answer, and the coach never answered it.
4. **A missing tag counts as an answer.** In about 1 reply in 7 the LLM leaves out the tag, and a reply interrupted while playing can lose its tag before it reaches the context. Both fall back to "answer", which means moving on one turn early, never looping.
