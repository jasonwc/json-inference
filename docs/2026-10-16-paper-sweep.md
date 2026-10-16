# Paper sweep 2026-10-16: every json-inference model through Byrdpip → paperd → Paper (staging, gateway `bots`, backend `json-sparks`)

Per model: `inference up`, served name checked on /v1/models, then 3 Hermes turns (`-m <served name>`):
(1) "which model are you, and 17×23?" (expect 391), (2) Home Assistant weather (tool use),
(3) train arrival 9:40 + 2h35 + 15 min delay (expect 12:30). "Captured" = turn's model calls reached
json-lab's public front from Paper's staging NAT (34.225.25.29) with the token accepted (auth=1, 200).
"Fallback" = Hermes used the local inference.json.lab instead (uncaptured).

| Definition | Served name | Started (load) | UTC window | Captured (turns) | Fallback calls | Turn latency (s) | Weather tool calls | Correct (/2: 391, 12:30) | Notes |
|---|---|---|---|---|---|---|---|---|---|
