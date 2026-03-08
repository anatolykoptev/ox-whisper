#!/usr/bin/env python3
"""WebSocket streaming test for ox-whisper /v1/listen endpoint.

Usage:
    python3 tests/ws_test.py [wav_file] [host:port]

Requires: pip install websockets
"""
import asyncio
import json
import sys
import wave

import websockets


async def test_ws(wav_path: str, host: str = "localhost", port: int = 8092):
    uri = f"ws://{host}:{port}/v1/listen?language=en&interim_results=true"
    print(f"Connecting to {uri}")
    print(f"Audio: {wav_path}")

    with wave.open(wav_path, "rb") as wf:
        assert wf.getframerate() == 16000, f"Expected 16kHz, got {wf.getframerate()}"
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()}"
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        duration_s = n_frames / 16000
        print(f"Duration: {duration_s:.1f}s, {len(raw)} bytes")

    async with websockets.connect(uri) as ws:
        # First message: Metadata
        meta = json.loads(await ws.recv())
        print(f"\n[Metadata] request_id={meta.get('request_id', '?')[:8]}... model={meta.get('model')}")
        assert meta["type"] == "Metadata", f"Expected Metadata, got {meta['type']}"

        # Send audio in 100ms chunks (3200 bytes for s16le @ 16kHz)
        chunk_size = 3200
        chunks_sent = 0
        for i in range(0, len(raw), chunk_size):
            await ws.send(raw[i:i + chunk_size])
            chunks_sent += 1
            await asyncio.sleep(0.02)

        print(f"Sent {chunks_sent} chunks ({len(raw)} bytes)")

        # Finalize
        await ws.send(json.dumps({"type": "Finalize"}))

        # Collect results
        results = []
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                t = msg.get("type", "?")
                if t == "Results":
                    alt = msg.get("channel", {}).get("alternatives", [{}])[0]
                    text = alt.get("transcript", "")[:80]
                    flag = "FINAL" if msg.get("is_final") else "interim"
                    print(f"  [{flag}] {text}")
                elif t == "SpeechStarted":
                    print(f"  [SpeechStarted] at {msg.get('timestamp_s', 0):.2f}s")
                elif t == "Error":
                    print(f"  [Error] {msg.get('message')}")
                results.append(msg)
                if msg.get("from_finalize"):
                    break
            except asyncio.TimeoutError:
                print("  (timeout waiting for more messages)")
                break

        # CloseStream
        await ws.send(json.dumps({"type": "CloseStream"}))
        try:
            close_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert close_msg["type"] == "CloseStream"
            print("\n[CloseStream] confirmed")
        except (asyncio.TimeoutError, Exception):
            pass

        # Summary
        finals = [
            r["channel"]["alternatives"][0]["transcript"]
            for r in results
            if r.get("type") == "Results" and r.get("is_final")
        ]
        interims = sum(1 for r in results if r.get("type") == "Results" and not r.get("is_final"))
        print(f"\nFinal transcript: {' '.join(finals)}")
        print(f"Messages: {len(results)} total, {len(finals)} final, {interims} interim")


if __name__ == "__main__":
    wav = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test.wav"
    addr = sys.argv[2] if len(sys.argv) > 2 else "localhost:8092"
    host, port = addr.split(":") if ":" in addr else (addr, "8092")
    asyncio.run(test_ws(wav, host, int(port)))
