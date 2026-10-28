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

Current step:
{step}
```

## Step instructions (one goes into `{step}`)

**Greeting** (no question asked yet):
```text
The session is starting. Greet the candidate in one short sentence, say this is a four question practice interview, then ask question 1 using its exact wording: "Tell me about a project you're proud of."
```

**First reply to question N** (the LLM chooses follow-up, nudge or move on):
```text
You asked question {n}. The candidate has just answered it. Decide:
- If the answer was vague, very short, or missing a concrete detail, ask ONE short follow-up about it.
- If the answer was empty, off topic, or they said they don't know, give ONE gentle nudge.
- If they asked you something instead of answering, answer it in one sentence, then ask question {n} again.
- Otherwise, react briefly to what they said and move on: {next_action}
```

**Second reply to question N** (follow-up used up, must move on):
```text
You asked question {n} and already used your one follow-up or nudge. Do not ask anything more about it. React briefly to what they said, then {next_action}
```

`{next_action}` is either the next question:
```text
ask question {n} using its exact wording: "{q}"
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

- **Code decides where we are and what is allowed.** `current_step()` scans the conversation:
  - The last agent message containing the end of a question's exact wording ("…do differently now") sets the current question. A question cut off by an interruption doesn't count. A question asked again resets the count.
  - Then it counts user replies since that question was asked. Several user messages in a row count as one reply, because Smart Turn can split one answer at sentence breaks.
  - One reply means the LLM may use its single follow-up or nudge, or move on. Two or more means it must move on.

  So order is fixed and follow-ups are capped at one per question, whatever the LLM "wants".
- **The LLM decides the judgement call.** It chooses whether this answer needs a follow-up (vague or missing detail), a nudge (empty, off topic, "I don't know"), or no further question, using the criteria in the first-reply step.
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

## Known failure cases

1. **Paraphrased question → asked twice.** Detection relies on the exact question wording. If the LLM rephrases ("What's a project you're really proud of?"), the director thinks question 1 was never asked and tells the LLM to ask it again.
2. **Follow-up that sounds like a later question.** A follow-up on question 2 ending "…how did you know it was working well?" matches question 3's wording, so the director skips ahead and question 3 is never asked as such.
3. **A question split by a sentence-break turn end.** In v3's run, Smart Turn ended the turn after "Sorry. Quick question.", before the actual question. The coach replied "Sure, what's your quick question?" and asked question 4 in the same breath. The candidate's real question then arrived merged with their answer, and the coach never answered it.
