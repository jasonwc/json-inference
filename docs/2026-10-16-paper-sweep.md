# Paper sweep 2026-10-16: every json-inference model through Byrdpip → paperd → Paper (staging, gateway `bots`, backend `json-sparks`)

Per model: `inference up`, served name checked on /v1/models, then 3 Hermes turns (`-m <served name>`):
(1) "which model are you, and 17×23?" (expect 391), (2) Home Assistant weather (tool use),
(3) train arrival 9:40 + 2h35 + 15 min delay (expect 12:30). "Captured" = turn's model calls reached
json-lab's public front from Paper's staging NAT (34.225.25.29) with the token accepted (auth=1, 200).
"Fallback" = Hermes used the local inference.json.lab instead (uncaptured).

| Definition | Served name | Started (load) | UTC window | Captured (turns) | Fallback calls | Turn latency (s) | Weather tool calls | Correct (/2: 391, 12:30) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-v4.1-flash-exl3 | DeepSeek-v4.1-Flash-EXL3 ✓ | yes (588 s) | 19:21:25–19:22:22 | 0/3: all 3 reached the backend via Paper (auth=1) but got **HTTP 400** | 3 → 404 (fallback asks for deepseek-v4-flash-vision-exp, which a strict vLLM rejects) | 16–23 | 0 | 0/2 (no answer) | 400 cause not in vLLM log; tools payload and context (600k) ruled out; Paper should hold the failed requests |
| mimo-v2.6-flash | MiMo-v2.6-Flash ✓ | yes (442 s) | 19:31:40–19:33:40 | **3/3** (16 model calls, all 200 auth=1) | 0 | 28 / 42 / 50 | 10 | 2/2 | weather: reported current conditions only ("no daily forecast"), missed sensor.forecast_today; also used 10 tool calls on the arithmetic turn |
| mimo-v2.6-flash-dflash | MiMo-v2.6-Flash ✓ (same name as above) | yes (449 s) | 19:42:24–19:44:18 | **3/3** (13 calls, all 200 auth=1) | 0 | 43 / 41 / 30 | 7 | 2/2 | tried terminal/execute_code for arithmetic (blocked by the sandbox policy, as designed); same served name as mimo-v2.6-flash, so Paper sessions are told apart only by the time window |
| qwen3.8-27b-nvfp4 | qwen3.8-27b-nvfp4 ✓ | yes (232 s) | 19:49:31–19:53:26 | **3/3** (20 calls, all 200 auth=1) | 0 | 58 / 77 / 100 | 14 | 2/2 | slowest per turn; weather: current conditions only; verified the arithmetic with a datetime calc. Two other 404s in this window were curl via Paper (not Byrdpip) asking for a model not loaded |
| qwen3.8-flash-next-nvfp4 | qwen3.8-flash-next ✓ | yes (758 s) | 20:07:30–20:09:34 | **3/3** (17 calls, all 200 auth=1) | 0 | 46 / 45 / 33 | 8 | 2/2 | ran awk via terminal to verify 17×23; weather: current conditions only |
