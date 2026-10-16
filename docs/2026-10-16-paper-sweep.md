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
| qwen3.8-flash-next-tensorfold | qwen3.8-flash-next-tensorfold ✓ | yes (96 s) | 20:12:24–20:14:03 | **3/3** (22 calls: 19 × 200, 3 × **400** that Hermes retried successfully) | 0 | 27 / 41 / 31 | 14 | 2/2 | fastest load; no repeat of the overnight 42-call loop (14 this time); weather: current conditions only |
| gpt-oss-120b | gpt-oss-120b ✓ (routed to json-sparks, no ambiguity) | yes (465 s) | 20:22:52–20:25:33 | **3/3** (21 calls: 18 × 200, 3 × **400** retried successfully) | 0 | 41 / 82 / 38 | 16 | 2/2 | weather: reported the "Forecast Home" entity |
| gpt-oss-120b-tp2 | gpt-oss-120b-tp2 ✓ | yes (148 s) | 20:28:52–20:31:02 | **3/3** (21 calls: 18 × 200, 3 × **400** retried successfully) | 0 | 40 / 56 / 34 | 16 | 2/2 | weather: reported the "Forecast Home" entity |
| deepseek-v4-flash-dspark (restore) | deepseek-v4-flash-vision-exp ✓ | yes (540 s) | 20:41:31–20:44:06 | **3/3** (18 calls, all 200 auth=1) | 0 | 36 / 54 / 65 | 10 | 2/2 | only model with no 400s at all |

Final state (20:45 UTC): deepseek-v4-flash-vision-exp serving; Byrdpip via paperd captured; HA voice LLM answers;
Osty and Byrdpip sandboxes active; alert silences expired; no alerts firing.

## Findings

- **Capture works for 8 of 9 definitions.** Every turn on MiMo (both), Qwen3.8-27B, Qwen3.8-Flash-Next
  (vLLM and TensorFold), gpt-oss-120b (both) and DeepSeek V4 Flash went Byrdpip → paperd → Paper →
  json-sparks with the token accepted, with no fallback. `gpt-oss-*` routed to json-sparks without ambiguity.
- **DeepSeek v4.1 EXL3 failed.** All 3 turns' Paper requests reached the backend and got HTTP 400.
  Hermes then fell back to the local endpoint, but the fallback asks for `deepseek-v4-flash-vision-exp`
  by name and this vLLM rejects other names (404), so no answers. The 400's cause isn't in the vLLM log.
  A plain request, a tools payload and context size (600k) all work directly.
- **Recurring 400s, about one per turn**, on TensorFold and both gpt-oss builds (3 × 400 among ~21 calls each).
  They are fast, 121-byte errors from upstream, sent by Hermes' OpenAI client, and Hermes retries them
  successfully. MiMo, Qwen 27B and Qwen Flash-Next vLLM showed none, and DeepSeek DSpark none.
- **The weather turn:** no model used the `sensor.forecast_today` high/low; all reported current conditions
  (7–16 tool calls). Byrdpip's briefing prompt names the sensor explicitly, so the briefing is fine, but
  a free-form "forecast" question misses it. Worth a hint in AGENTS.md.
- **Arithmetic and the time question:** every model that answered was correct (391; 12:30). Several tried
  terminal/execute_code first, which the sandbox policy blocks by design.

## For the Paper agent (sessions under jason-hermes-staging, gateway bots → json-sparks)

| UTC window (2026-10-16) | Served model name | Expected in Paper |
|---|---|---|
| 19:21:25–19:22:22 | DeepSeek-v4.1-Flash-EXL3 | 3 failed requests (400 from backend) |
| 19:31:40–19:33:40 | MiMo-v2.6-Flash | 3 sessions, 16 calls (mimo-v2.6-flash) |
| 19:42:24–19:44:18 | MiMo-v2.6-Flash | 3 sessions, 13 calls (mimo-v2.6-flash-dflash build) |
| 19:49:31–19:53:26 | qwen3.8-27b-nvfp4 | 3 sessions, 20 calls (plus 2 curl 404s at ~19:50 not from Byrdpip) |
| 20:07:30–20:09:34 | qwen3.8-flash-next | 3 sessions, 17 calls |
| 20:12:24–20:14:03 | qwen3.8-flash-next-tensorfold | 3 sessions, 22 calls (3 × 400, retried) |
| 20:22:52–20:25:33 | gpt-oss-120b | 3 sessions, 21 calls (3 × 400, retried) |
| 20:28:52–20:31:02 | gpt-oss-120b-tp2 | 3 sessions, 21 calls (3 × 400, retried) |
| 20:41:31–20:44:06 | deepseek-v4-flash-vision-exp | 3 sessions, 18 calls |
