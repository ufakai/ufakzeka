"""Re-score a saved scenario run against the current suite checks, from its stored transcripts (no model).

  python scripts/rescore_scenarios.py --suite scenarios.json --run evals/scenarios_v40.json

Used when a check regex is corrected: the transcripts are the model's answers, only the rules changed.
"""
import argparse, json
from collections import defaultdict
from run_scenarios import check_turn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--suite", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", default="")
    a = ap.parse_args()
    # match by the conversation text: ids were counters before 2026-09-01 and renumbered on insertion
    suite = {json.dumps(s["turns"], ensure_ascii=False): s for s in json.load(open(a.suite, encoding="utf-8"))}
    run = json.load(open(a.run, encoding="utf-8"))
    per = defaultdict(lambda: [0, 0]); failures = []
    for t in run["transcripts"]:
        s = suite.get(json.dumps([m["content"] for m in t["messages"][0::2]], ensure_ascii=False))
        if s is None:
            continue
        msgs = t["messages"]
        for i, turn in enumerate(s["turns"]):
            checks = s["checks"].get(str(i))
            if checks is None or 2 * i + 1 >= len(msgs):
                continue
            ans = msgs[2 * i + 1]["content"]
            fails = check_turn(ans, [tuple(c) for c in checks], {**s.get("meta", {}), "family": s["family"], "user": turn, "prev_answers": [x["content"] for x in msgs[:2 * i + 1] if x["role"] == "assistant"]})
            per[s["family"]][1] += 1
            if fails:
                failures.append({"id": s["id"], "family": s["family"], "seed": t["seed"], "turn": i, "user": turn, "answer": ans[:400], "fails": fails})
            else:
                per[s["family"]][0] += 1
    ok = sum(v[0] for v in per.values()); n = sum(v[1] for v in per.values())
    print(f"{a.run}: {ok}/{n} ({100 * ok / max(n, 1):.1f}%) on the current suite")
    for f, (k, m) in sorted(per.items(), key=lambda x: x[1][0] / max(x[1][1], 1)):
        print(f"  {f:16}{100 * k / max(m, 1):6.1f}% {m:5}")
    if a.out:
        json.dump({**run, "summary": {f: {"pass": v[0], "n": v[1]} for f, v in per.items()}, "overall": {"pass": ok, "n": n}, "failures": failures},
                  open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
