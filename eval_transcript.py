"""Score a session transcript's follow-ups and feedback with an LLM judge.

Usage:  python eval_transcript.py sessions/<id>/transcript.txt

Uses the same Groq key as the bot. The judge is a different prompt from the
coach, but the same model family, so treat scores as a rough signal for
comparing prompt versions, not ground truth.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

JUDGE_PROMPT = """You are reviewing a transcript of a spoken mock interview between an AI coach and a candidate.
The coach must ask four fixed questions in order, react to what the candidate actually said, ask at most one
follow-up or nudge per question, and end with two or three sentences of feedback.

Score each item from 1 (poor) to 5 (excellent) and give a one-sentence reason:
- order: were the four questions asked in order, none skipped or repeated?
- grounded_reactions: do the coach's reactions and follow-ups refer to specific things the candidate said?
- follow_up_judgement: were follow-ups used where the answer was thin, and skipped where it was already good?
- feedback_specific: does the closing feedback name a real strength and a concrete improvement from this session?
- voice_style: are the coach's turns short, spoken-style, free of lists and markdown?

Reply with JSON only, shaped like:
{"order": {"score": 5, "reason": "..."}, "grounded_reactions": {...}, "follow_up_judgement": {...},
 "feedback_specific": {...}, "voice_style": {...}}"""


def main():
    transcript = Path(sys.argv[1]).read_text(encoding="utf-8")
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=os.getenv("JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")),
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": transcript},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    scores = json.loads(response.choices[0].message.content)
    for name, item in scores.items():
        print(f"{name:20} {item['score']}/5  {item['reason']}")
    print(f"{'mean':20} {sum(i['score'] for i in scores.values()) / len(scores):.1f}/5")


if __name__ == "__main__":
    main()
