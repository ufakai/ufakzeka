"""Turn eval summaries (json blocks in a log, or summary.json files) into one markdown table."""

import json
import re
import sys

KEYS = ["hellaswag_tr", "arc_tr_challenge", "arc_tr_easy", "xcopa_tr", "belebele_tur_Latn", "turblimp_core", "turkishmmlu"]


def rows_from_log(path):
    text = open(path).read()
    out = {}
    for name, block in re.findall(r"=== (\S+)\n(.*?)(?=\n=== |\Z)", text, re.S):
        m = re.search(r"\n\{\n.*?\n\}\n", block, re.S)
        if not m:
            continue
        out[name] = json.loads(m.group(0))
    return out


def fmt(res):
    def g(task, key="acc_norm,none", alt="acc,none"):
        d = res.get(task, {})
        v = d.get(key, d.get(alt))
        return f"{100*v:.1f}" if isinstance(v, float) else "-"
    tb = [v for k, v in res.items() if k.startswith("turblimp_") and isinstance(v, dict) and "acc,none" in v]
    tb_avg = f"{100*sum(x['acc,none'] for x in tb)/len(tb):.1f}" if tb else g("turblimp_core", "acc,none")
    mm = [v for k, v in res.items() if k.startswith("turkishmmlu_") and isinstance(v, dict) and "acc,none" in v]
    mm_avg = f"{100*sum(x['acc,none'] for x in mm)/len(mm):.1f}" if mm else g("turkishmmlu", "acc,none")
    return [g("hellaswag_tr"), g("arc_tr_challenge"), g("arc_tr_easy"), g("xcopa_tr", "acc,none"), g("belebele_tur_Latn", "acc,none"), tb_avg, mm_avg]


if __name__ == "__main__":
    allres = rows_from_log(sys.argv[1])
    print("| model | HellaSwag-tr (acc_norm) | ARC-c-tr (acc_norm) | ARC-e-tr (acc_norm) | XCOPA-tr | Belebele-tur | TurBLiMP (avg) | TurkishMMLU (avg) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, res in allres.items():
        print("| " + name + " | " + " | ".join(fmt(res)) + " |")
