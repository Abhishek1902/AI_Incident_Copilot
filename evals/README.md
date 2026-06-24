# Eval set

Eval corpus uses synthetic incident data; see [`app/data/sample_incidents.json`](../app/data/sample_incidents.json).

[`eval_set.jsonl`](eval_set.jsonl) holds the hand-graded questions; each `ground_truth_event_ids` entry points at events in that synthetic fixture. To run any question end-to-end, seed the DB first with `python scripts/seed.py`.
