"""Check a live session's logs against the fixes found in simulated testing.

Usage:  python tests/check_session.py [sessions/<id>] [bot.log]
Defaults to the newest session folder and ./bot.log.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interview import QUESTIONS, asked_question  # noqa: E402
from session_log import summarise  # noqa: E402


def main():
    session = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("sessions").glob("2*"))[-1]
    bot_log = Path(sys.argv[2] if len(sys.argv) > 2 else "bot.log").read_text(encoding="utf-8", errors="replace")
    lines = (session / "transcript.txt").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in (session / "latency.jsonl").read_text(encoding="utf-8").splitlines()]
    turns = [(m.group(1), m.group(2)) for l in lines if (m := re.match(r"\[\s*[\d.]+s\] (\w+): (.*)", l))]

    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    # Questions: each fully asked, in order (fixes 1 and 2).
    asked = []
    for who, text in turns:
        if who == "Coach" and "[interrupted by user]" not in text and (n := asked_question(text)) and n not in asked:
            asked.append(n)
    check("all 4 questions fully asked, in order", asked == [1, 2, 3, 4], f"order {asked}")

    # Follow-ups possible (fix 1): a coach question that isn't one of the four.
    follow_ups = [t for w, t in turns if w == "Coach" and "?" in t and not asked_question(t) and "end of our practice" not in t]
    check("at least one follow-up or nudge asked", bool(follow_ups), follow_ups[0][:90] if follow_ups else "none")

    # Question 4 answered before feedback (fix 2).
    q4 = next((i for i, (w, t) in enumerate(turns) if w == "Coach" and asked_question(t) == 4 and "[interrupted" not in t), None)
    answered = q4 is not None and any(w == "User" for w, _ in turns[q4 + 1 :])
    check("question 4 answered before feedback", answered)

    closing = any(w == "Coach" and "end of our practice session" in t.lower() for w, t in turns)
    check("closing line spoken and session ended", closing and "Closing line spoken, ending session" in bot_log)

    # Scenario B: no coach reply between the paused fragments of the first answer.
    first_answer = next((i for i, (w, t) in enumerate(turns) if w == "User"), None)
    next_coach = next((t for w, t in turns[first_answer:] if w == "Coach"), "") if first_answer is not None else ""
    check("scenario B: coach waited through the pause", "Take your time" not in next_coach and "[interrupted" not in next_coach,
          f"first coach reply after answer: {next_coach[:80]}")

    # Fix 3: no phantom empty turns.
    empty = bot_log.count("Empty user turn")
    check("no phantom empty user turns", empty == 0, f"{empty} found")

    # Scenario C: the coach was interrupted at least once.
    interrupted_rows = [r["turn"] for r in rows if r["interrupted"]]
    check("scenario C: an interruption was recorded", bool(interrupted_rows) and any("[interrupted by user]" in l for l in lines),
          f"turns {interrupted_rows}")

    # Fix 4: every latency row is complete and in time order.
    keys = ("user_stopped_ms", "stt_final_ms", "llm_request_ms", "llm_first_token_ms", "tts_first_audio_ms")
    bad = [r["turn"] for r in rows if None in (vals := [r[k] for k in keys]) or vals[2:] != sorted(vals[2:]) or vals[0] > vals[4]]
    check("latency rows complete and ordered", rows != [] and not bad, f"{len(rows)} rows, bad turns {bad}")

    # With a simulator result, check the coach never replied in the middle of
    # an answer: the first and last words of each answer must fall between
    # the same two coach lines.
    sims = sorted(Path("sessions").glob("sim-*.json"))
    if sims and sims[-1].stat().st_mtime >= (session / "transcript.txt").stat().st_mtime - 600:
        said = [t["said"] for t in json.loads(sims[-1].read_text(encoding="utf-8"))["turns"] if t.get("said")]
        # Every user word, with how many coach lines came before it.
        words, coach_before, coach_lines = [], [], 0
        for who, text in turns:
            if who == "Coach":
                coach_lines += 1
                continue
            for w in re.findall(r"[a-z]+", text.lower()):
                words.append(w)
                coach_before.append(coach_lines)

        def locate(phrase, last):
            target = re.findall(r"[a-z]{3,}", phrase.lower())
            target = target[-3:] if last else target[:3]
            for i in (range(len(words) - 1, -1, -1) if last else range(len(words))):
                window = [w for w in words[i : i + 8] if len(w) >= 3][: len(target)]
                if window == target:
                    return i
            return None

        mid_answer = []
        for text in said:
            start, end = locate(text, last=False), locate(text, last=True)
            if start is not None and end is not None and coach_before[start] != coach_before[end]:
                mid_answer.append(text[:50])
        check("coach never replied mid-answer", not mid_answer, f"{len(said)} answers checked; split: {mid_answer}")

    gaps = sorted(r["tts_first_audio_ms"] - r["user_stopped_ms"] for r in rows)
    check("worst user-stopped -> first audio under 2.5 s", bool(gaps) and gaps[-1] < 2500, f"gaps {gaps}")

    print()
    print(summarise(rows))
    print(f"{sum(checks)}/{len(checks)} checks passed")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
