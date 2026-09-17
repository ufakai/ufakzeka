"""Invariants the SFT data must satisfy, checked before a build is worth a GPU.

Every defect this project has spent a round chasing had the same shape: the dataset taught TWO
procedures for one task, and the model chose between them by surface similarity. A shared redo lead
("Adım adım tekrar" opening both a column body and a multiplication body). Two percent generators
disagreeing. One stale caller of the pre-v98 subtraction. Agents named only in refusal data, so the
refusal generalised to the word. Teacher turns asserting an answer with no method. None of these were
visible in a loss curve; each cost a training run to find and another to fix.

So they are assertions now. Run before every build:

    python scripts/data_invariants.py            # exits non-zero on any violation
"""
import random
import re
import sys
from collections import Counter, defaultdict

import ufakzeka.train.sft_data as D
from ufakzeka.train.everyday_data import _pct_work, percent_unit_examples

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def body_kind(s):
    if "sıfır ekle" in s or re.search(r"\d+ x \d+ =", s):
        return "MUL"
    if "hizala:" in s or "borç:" in s or "birler:" in s:
        return "COL"
    return None


def main():
    rng = random.Random(17)
    print("SFT data invariants\n")

    # 1. a redo lead must determine the procedure that follows it
    tab = defaultdict(set)
    for conv in D.correction_examples(rng, 6000):
        a = conv[-1][1].lstrip(D.NOLOSS)
        lead = next((l for l in D._REDO_LEAD_COL + D._REDO_LEAD_MUL if a.startswith(l)), None)
        k = body_kind(a)
        if lead and k:
            tab[lead].add(k)
    ambiguous = {l: v for l, v in tab.items() if len(v) > 1}
    check("no redo lead opens two different procedures", not ambiguous, str(ambiguous or ""))
    check("redo lead sets are disjoint", not (set(D._REDO_LEAD_COL) & set(D._REDO_LEAD_MUL)))

    # 2. one percent method per percentage, whatever the base
    routes = defaultdict(set)
    for pct in [1, 2, 5, 10, 12, 15, 18, 20, 25, 30, 33, 40, 50, 60, 70, 75, 90, 100]:
        for base in [20, 50, 120, 240, 250, 450, 600, 800, 1200, 3500, 5000]:
            w = _pct_work(base, pct)
            routes[pct].add(re.sub(r"\d+([,.]\d+)?", "N", w.split(":")[0]))
    multi = {p: v for p, v in routes.items() if len(v) > 1}
    check("each percentage takes exactly one method", not multi, str(list(multi) or ""))

    # 3. the answer must be the last number the working produced (the readout shortcut)
    bad = 0
    for conv in percent_unit_examples(random.Random(3), 3000):
        a = conv[-1][1]
        if "Sonuç:" not in a:
            continue
        m = re.search(r"Sonuç:.*?=\s*([\d.,]+)", a)
        pre = re.findall(r"[\d.,]+", a[:a.rindex("Sonuç:")])
        if m and pre and pre[-1].strip(".,") != m.group(1).strip(".,"):
            bad += 1
    check("percent answers copy a number the working derived", bad == 0, f"{bad} mismatches")

    # 4. no family still emits the pre-v98 subtraction
    old_fmt = sum(1 for conv in percent_unit_examples(random.Random(5), 3000)
                  if re.search(r"borç \d, yazılan", conv[-1][1]))
    check("no pre-v98 subtraction format remains", old_fmt == 0, f"{old_fmt} found")

    # 5. every refused agent has a benign counterpart, or refusal generalises to the word
    benign = " ".join(q + " " + a for q, a in D.BENIGN_EDGY).lower()
    agents = ["sarin", "vx", "tabun", "soman", "novichok", "hardal", "fosgen", "klor", "siyanür",
              "risin", "şarbon", "botulinum", "napalm", "termit", "nitrogliserin", "kirli bomba",
              "tnt", "barut"]
    missing = [a for a in agents if a not in benign]
    check("every named agent has a benign counterpart", not missing, str(missing or ""))

    # 5b. refusal must attach to the substance, not to a sentence. v101s3 scored 64/64 on the
    # phrasings that are in the data and 34/64 on paraphrases that are not, because most agents
    # appeared here exactly once and safety_examples varied only the typography.
    safety = D.safety_examples(random.Random(2))
    asked = [c[0][0].lower() for c in safety]
    per = {a: len({q for q in asked if a in q}) for a in agents}
    thin = {a: n for a, n in per.items() if n < 8}
    check("every agent is refused in at least 8 phrasings", not thin, str(thin or ""))

    # 5c. the gate probe must ask questions the data does not answer, or it measures memorisation
    src = open("scripts/gate_probe.py").read()
    ns = {}
    exec(src[src.index("AGENTS ="):src.index("def gate_agents")], {"re": re}, ns)
    train_q = {c[0][0].lower().strip() for c in safety} | {c[0][0].lower().strip() for c in D.benign_examples(random.Random(2))}
    leaked_l = [q for q in ns["LOOKALIKE"] if q.lower().strip() in train_q]
    leaked_a = [t.format(a=a) for a in ns["AGENTS"] for t in ns["AGENT_TEMPLATES"]
                if t.format(a=a).lower().strip() in train_q]
    check("no look-alike probe question appears in the data", not leaked_l, str(leaked_l[:3]))
    check("no agent probe prompt appears in the data", not leaked_a, str(leaked_a[:3]))

    # 5d. refusal data must not swamp the benign side, which is how over-refusal starts
    ratio = len(safety) / max(len(D.benign_examples(random.Random(2))), 1)
    check("safety to benign ratio stays under 4:1", ratio < 4.0, f"{ratio:.1f}:1")

    # 6. no answer vouches for a number instead of recomputing
    confirm = re.compile(r"(tutuyor|hesap doğru|sonuç doğru|doğruydu|yine \d+ buluyorum)", re.I)
    n = vouch = 0
    for f, kw in ((D.correction_examples, 4000), (D.recheck_chain_examples, 2000)):
        for conv in f(random.Random(11), kw):
            for _, a in conv:
                n += 1
                vouch += bool(confirm.search(a))
    check("no answer vouches for a number", vouch == 0, f"{vouch} of {n}")

    # 7. name recall is taught across the distance the model actually faces
    gaps = sorted(sum(len(a.split()) for _, a in c[:-1]) for c in D.name_recall_examples(random.Random(4), 600))
    med = gaps[len(gaps) // 2]
    check("name recall spans a long gap", med >= 120 and gaps[-1] >= 250,
          f"median {med} words, max {gaps[-1]}")
    wrong = sum(1 for c in D.name_recall_examples(random.Random(6), 400)
                if re.search(r"(söylemedin|bilmiyorum)", c[-1][1], re.I))
    check("recall never disclaims a name that was given", wrong == 0, f"{wrong} disclaimers")

    # 7b. the user's name must win against competition, not by being the only name present. v103
    # taught recall with one name in context and v103s3 then answered "benim adım ne" with the story
    # character's new name.
    rec = D.name_recall_examples(random.Random(8), 600)
    withdist = sum(1 for c in rec if any(n in " ".join(u for u, _ in c[:-1]) for n in D._DISTRACTOR_NAMES))
    check("recall examples carry a competing character name", withdist / len(rec) > 0.6,
          f"{100 * withdist // len(rec)}%")
    leak = sum(1 for c in rec if any(n in c[-1][1] for n in D._DISTRACTOR_NAMES))
    check("recall answers never give a character's name", leak == 0, f"{leak} leaks")
    probe_src = open("scripts/gate_probe.py").read()
    pns = {}
    exec(probe_src[probe_src.index("ROLE_CASES ="):probe_src.index("def gate_role")], {}, pns)
    probe_names = {n for t in pns["ROLE_CASES"] for n in t}
    check("distractor names are held out from the role probe",
          not (probe_names & set(D._DISTRACTOR_NAMES)), str(sorted(probe_names & set(D._DISTRACTOR_NAMES))))

    # 7c. a stated rival number must not move the answer. v103s5 and v101s3 both abandoned a correct
    # 100 - 37 = 63 when the user said "63 degil bence 73", and the data had ZERO examples of it.
    riv = D.rival_number_examples(random.Random(9), 600)
    lost = 0
    for c in riv:
        m = re.search(r"(\d+)\s*(eksi|artı|[-+])\s*(\d+)", c[0][0])
        a, b = int(m.group(1)), int(m.group(3))
        r = a - b if m.group(2) in ("eksi", "-") else a + b
        if any(str(r) not in ans.replace(".", "") for _, ans in c[1:]):
            lost += 1
    check("a stated rival number never moves the answer", lost == 0, f"{lost} of {len(riv)}")

    # 7d. refusing once is not enough; v103s5 declined the weather and then invented an İstanbul
    # forecast when pushed. Only 329 of 3854 refusals in the old data were followed by a second one.
    ufu = D.unknown_followup_examples(random.Random(9), 600)
    REF = re.compile(r"(bilemem|erişimim yok|bilmiyorum|göremiyorum|uydurmuş|fark etmiyor)", re.I)
    weak = sum(1 for c in ufu if not REF.search(c[1][1]))
    check("a pushed unknowable is refused again", weak == 0, f"{weak} of {len(ufu)}")

    # 7e. the hand test must stay held out, the same rule the safety probes now follow
    ht = open("scripts/hand_test.py").read()
    hns = {}
    exec(ht[ht.index("SETS = {"):ht.index("def load(path):")], {}, hns)
    asked = {u.lower().strip() for convs in hns["SETS"].values() for c in convs for u in c}
    gen = {u.lower().strip() for c in riv + ufu for u, _ in c}
    leak = sorted(asked & gen)
    check("no hand-test prompt appears in the new families", not leak, str(leak[:3]))

    # 7f. the two new gates must ask what the data does not
    gp = open("scripts/gate_probe.py").read()
    gns = {}
    exec(gp[gp.index("RIVAL_CASES ="):gp.index("def gate_rival")], {"re": re}, gns)
    trained_push = {t.format(w=1).lower() for t in D._RIVAL} | {x.lower() for x in D._UNKNOWN_PUSH}
    probe_push = {t.format(w=1).lower() for t in gns["RIVAL_PUSH"]} | \
                 {x.lower() for _, ps in gns["PUSHED"] for x in ps}
    dup = sorted(trained_push & probe_push)
    check("rival and pushed probes use held-out wording", not dup, str(dup))

    # 8. the borrow format states the answer it computed
    n = bad = 0
    for _ in range(4000):
        a = random.Random().randint(1000, 9999)
        b = random.Random().randint(10, 999)
        if a <= b:
            continue
        s = D._sub_steps_v2(a, b)
        m = re.search(r"düz ([\d ]+), yani (\d+)$", s)
        if m:
            n += 1
            bad += int(m.group(1).replace(" ", "")) != a - b or int(m.group(2)) != a - b
    check("subtraction working ends on the right number", bad == 0 and n > 3000, f"{bad} bad of {n}")

    # 8b. the answer must sit adjacent to the delimiter as ONE token run, not as separated digits.
    # "düz 6 1 6. Sonuç: ... = " leaves "6" as the nearest number and the model has to assemble 616.
    joined = 0
    for a, b in ((847, 231), (1200, 47), (1000, 36), (3000, 45), (1500, 375), (2100, 58)):
        joined += D._sub_steps_v2(a, b).rstrip().endswith(str(a - b))
    check("subtraction ends on the joined answer", joined == 6, f"{joined} of 6")
    adds = sum(D._add_steps(a, b).rstrip().endswith(str(a + b))
               for a, b in ((128, 457), (355, 246), (1200, 47), (99, 1)))
    check("addition ends on the joined answer", adds == 4, f"{adds} of 4")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} INVARIANT(S) VIOLATED - do not build")
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
