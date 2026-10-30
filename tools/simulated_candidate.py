"""Simulated candidate: talks to the running bot over WebRTC, like a browser would.

Start the bot first (python bot.py), then:
    python tools/simulated_candidate.py

The candidate's answers come from a Groq LLM playing an engineer, voiced by
Cartesia (or Deepgram when TTS_PROVIDER=deepgram). It acts out the three scenarios from the brief:
    A: normal answers
    B: first answer has a mid-sentence pause ("I worked on... um... <2 s> payments")
    C: on the third answer it talks over the coach's reply

It also measures, on the client side, the gap from "candidate audio ended" to
"coach audio arrived", and how fast the coach went quiet when interrupted.
Results go to sessions/sim-<timestamp>.json; the bot writes its own
latency.jsonl and transcript.txt as usual.
"""

import asyncio
import fractions
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame
from dotenv import load_dotenv

load_dotenv(override=True)

BOT_URL = os.getenv("SIM_BOT_URL", "http://localhost:7860")
VOICE_ID = os.getenv("SIM_VOICE_ID", "47c38ca4-5f35-497b-b1a3-415245fb35e1")
# Used instead of Cartesia when TTS_PROVIDER=deepgram, like the bot.
SIM_DEEPGRAM_VOICE = os.getenv("SIM_DEEPGRAM_VOICE", "aura-2-orion-en")
SAMPLE_RATE = 16000
CHUNK = SAMPLE_RATE // 50  # 20 ms
SPEECH_RMS = 300  # int16 RMS above this counts as the coach speaking
SILENCE_DONE_SECS = 1.2  # coach is done after this much silence
CLOSING = "end of our practice session"

CANDIDATE_PROMPT = """You are a software engineer in a spoken mock interview. Your project: a payments
reconciliation service you built at a fintech startup, matching bank settlement files against
ledger entries. Answer the interviewer's last message in two or three short spoken sentences.
Be specific and natural, plain text only, no lists. If the interviewer is wrapping up, just say thanks."""


class ScriptedMic(MediaStreamTrack):
    """An outgoing audio track that plays queued PCM in real time, else silence."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._buf = bytearray()
        self._pts = 0
        self._start: float | None = None
        self.last_audio_end: float | None = None
        self.drained = asyncio.Event()
        self.drained.set()

    def say(self, pcm: bytes):
        self._buf += pcm
        self.drained.clear()

    async def recv(self) -> AudioFrame:
        if self._start is None:
            self._start = time.time()
        # Pace frames at real time.
        wait = self._start + self._pts / SAMPLE_RATE - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        n = CHUNK * 2
        if self._buf:
            data = bytes(self._buf[:n]).ljust(n, b"\0")
            del self._buf[:n]
            if not self._buf:
                self.last_audio_end = time.time() + 0.02
                self.drained.set()
        else:
            data = b"\0" * n
        frame = AudioFrame.from_ndarray(np.frombuffer(data, np.int16).reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
        self._pts += CHUNK
        return frame


class CoachEar:
    """Watches the incoming audio track and tracks when the coach is speaking."""

    def __init__(self):
        self.speaking = False
        self.started_at: float | None = None
        self.last_voice: float = 0.0
        self.started = asyncio.Event()
        self.finished = asyncio.Event()

    async def listen(self, track):
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            pcm = frame.to_ndarray().astype(np.float32)
            rms = float(np.sqrt(np.mean(pcm**2)))
            now = time.time()
            if rms > SPEECH_RMS:
                self.last_voice = now
                if not self.speaking:
                    self.speaking = True
                    self.started_at = now
                    self.finished.clear()
                    self.started.set()
            elif self.speaking and now - self.last_voice > SILENCE_DONE_SECS:
                self.speaking = False
                self.started.clear()
                self.finished.set()

    def reset(self):
        self.started.clear()
        self.finished.clear()


async def tts(session: aiohttp.ClientSession, text: str) -> bytes:
    if os.getenv("TTS_PROVIDER", "cartesia").lower() == "deepgram":
        async with session.post(
            "https://api.deepgram.com/v1/speak",
            params={"model": SIM_DEEPGRAM_VOICE, "encoding": "linear16", "sample_rate": str(SAMPLE_RATE), "container": "none"},
            headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
            json={"text": text},
        ) as r:
            r.raise_for_status()
            return await r.read()
    async with session.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={"X-API-Key": os.environ["CARTESIA_API_KEY"], "Cartesia-Version": "2025-04-16"},
        json={
            "model_id": "sonic-3.6",
            "transcript": text,
            "voice": {"mode": "id", "id": VOICE_ID},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
            "language": "en",
        },
    ) as r:
        r.raise_for_status()
        return await r.read()


async def candidate_reply(session: aiohttp.ClientSession, history: list[dict]) -> str:
    async with session.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "reasoning_effort": "low",
            "temperature": 0.8,
            "messages": [{"role": "system", "content": CANDIDATE_PROMPT}, *history],
        },
    ) as r:
        r.raise_for_status()
        return (await r.json())["choices"][0]["message"]["content"].strip()


def latest_coach_line(since: float) -> str:
    """Last coach line from the newest bot session transcript."""
    sessions = sorted(p for p in Path("sessions").glob("2*") if p.stat().st_mtime >= since)
    if not sessions or not (sessions[-1] / "transcript.txt").exists():
        return ""
    lines = (sessions[-1] / "transcript.txt").read_text(encoding="utf-8").splitlines()
    coach = [l for l in lines if "] Coach: " in l]
    return re.sub(r"^.*\] Coach: ", "", coach[-1]).replace(" [interrupted by user]", "") if coach else ""


def silence(secs: float) -> bytes:
    return b"\0" * int(SAMPLE_RATE * secs) * 2


async def main():
    started = time.time()
    mic, ear = ScriptedMic(), CoachEar()
    pc = RTCPeerConnection()
    pc.createDataChannel("chat")
    pc.addTrack(mic)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.ensure_future(ear.listen(track))

    await pc.setLocalDescription(await pc.createOffer())
    results = {"turns": [], "interruption": None}

    async with aiohttp.ClientSession() as http:
        async with http.post(
            f"{BOT_URL}/api/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ) as r:
            answer = await r.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
        print("connected, waiting for the coach to greet")

        await asyncio.wait_for(ear.started.wait(), 30)
        await asyncio.wait_for(ear.finished.wait(), 60)
        history: list[dict] = []

        for turn in range(1, 13):
            await asyncio.sleep(0.5)  # let the bot write its transcript line
            coach = latest_coach_line(started)
            print(f"\nCOACH: {coach}")
            if CLOSING in coach.lower():
                break
            history.append({"role": "user", "content": coach})
            # One line, so TTS doesn't insert a paragraph-length pause.
            reply = " ".join((await candidate_reply(http, history)).split())
            # Plain punctuation; Unicode dashes made the TTS pause oddly.
            reply = re.sub("[\u2010-\u2015]", "-", reply).replace("\u2019", "'")
            history.append({"role": "assistant", "content": reply})

            if turn == 1:
                # Scenario B: pause mid-answer.
                lead = "So, I worked on... um..."
                print(f"CANDIDATE (B, pause): {lead} <2s pause> {reply}")
                pcm = await tts(http, lead) + silence(2.0) + await tts(http, reply)
            else:
                print(f"CANDIDATE{' (C, will interrupt)' if turn == 3 else ''}: {reply}")
                pcm = await tts(http, reply)

            ear.reset()
            mic.say(pcm)
            await mic.drained.wait()
            user_end = mic.last_audio_end
            try:
                await asyncio.wait_for(ear.started.wait(), 20)
            except asyncio.TimeoutError:
                print("  !! coach did not reply within 20 s")
                results["turns"].append({"turn": turn, "said": reply, "client_gap_ms": None})
                continue
            gap = round((ear.started_at - user_end) * 1000)
            results["turns"].append({"turn": turn, "said": reply, "client_gap_ms": gap})
            print(f"  client-side gap, candidate stopped -> coach audio: {gap} ms")

            if turn == 3:
                # Scenario C: talk over the coach 1.5 s into its reply.
                await asyncio.sleep(1.5)
                # ear.speaking lags real silence by SILENCE_DONE_SECS, so check
                # that coach audio is actually arriving right now.
                if ear.speaking and time.time() - ear.last_voice < 0.3:
                    interjection = "Sorry, quick question, should I focus on the technical side or the business side?"
                    print(f"CANDIDATE (C, interrupting): {interjection}")
                    history.append({"role": "assistant", "content": interjection})
                    ear.finished.clear()
                    mic.say(await tts(http, interjection))
                    t0 = time.time()
                    await asyncio.wait_for(ear.finished.wait(), 30)
                    stop_ms = round((ear.last_voice - t0) * 1000)
                    results["interruption"] = {"coach_kept_talking_ms": stop_ms}
                    print(f"  coach audio stopped {stop_ms} ms after the candidate started talking")
                    await mic.drained.wait()
                    ear.reset()
                    await asyncio.wait_for(ear.started.wait(), 20)
                else:
                    print("  coach had already stopped; no interruption possible")
            await asyncio.wait_for(ear.finished.wait(), 90)

    await asyncio.sleep(2)
    await pc.close()
    gaps = [t["client_gap_ms"] for t in results["turns"] if t["client_gap_ms"] is not None]
    if gaps:
        results["client_gap_ms_sorted"] = sorted(gaps)
        print(f"\nclient-side gaps (ms): {sorted(gaps)}")
    out = Path("sessions") / f"sim-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
