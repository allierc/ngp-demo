#!/usr/bin/env python
"""Fill note/collision_audit.tex from figures/collision_audit.json and run pdflatex.

    python note/build_note.py

The note quotes about forty numbers.  Typing them would guarantee that some of
them stop matching the run that produced the figure, so the .tex carries \\NAME
placeholders and every one of them is resolved here from the json the audit
wrote.  An unresolved or unused placeholder is an error, not a warning.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSON = os.path.join(ROOT, "figures", "collision_audit.json")


def main():
    d = json.load(open(JSON))
    s, c, runs = d["settings"], d["content"], d["runs"]
    lams = sorted({r["lam"] for r in runs.values()}, reverse=True)
    lo, hi = lams[0], lams[-1]                       # 1.0 and 0.01
    mid = lams[1]

    def run(tag, lam):
        return runs[f"{tag}_lam{lam:g}"]

    def cost(reg, lam):                              # dB the collisions took
        return run("dense", lam)[f"psnr_{reg}"] - run("collide", lam)[f"psnr_{reg}"]

    hashed = [x for x in d["census"] if not x["dense"]]
    wdev = max(abs(x["weight_ratio"] - 1.0) for x in hashed)
    p1 = {f"{p['lo']:.1f}": p for p in d["perturb"][f"collide_lam{lo:g}"]}
    ph = {f"{p['lo']:.1f}": p for p in d["perturb"][f"collide_lam{hi:g}"]}

    def share_a(p):
        return p["drop_a"] / max(1e-9, p["drop_a"] + p["drop_b"])

    rows = " \\\\\n".join(
        f"{l:g} & {run('collide', l)['psnr_a']:.2f} & {run('collide', l)['psnr_b']:.2f} & "
        f"{run('dense', l)['psnr_a']:.2f} & {run('dense', l)['psnr_b']:.2f} & "
        f"{cost('b', l):.2f} dB" for l in lams) + " \\\\"

    v = {
        "CHKBLOCK": f"{s['block']}",
        "PXA": f"{c['px_a']:,}", "PXB": f"{c['px_b']:,}",
        "DETA": f"{c['detail_a']:.4f}", "DETB": f"{c['detail_b']:.4f}",
        "WEIGHTDEV": f"{wdev*100:.1f}\\%",
        "SMALLT": f"{s['log2_small']}", "LARGET": f"{s['log2_large']}",
        "HASHEDLEVELS": f"{run('collide', lo)['n_hashed']}",
        "NLEVELS": f"{s['encoder']['n_levels']}",
        "IMSIZE": "$" + s["image"].replace("x", r"\times", 1).split("x")[0] + "$",
        "STEPS": f"{s['steps']:,}", "LR": f"$10^{{{round(-abs(__import__('math').log10(s['lr']))):g}}}$",
        "BATCH": f"{s['batch']:,}",
        "DENSESHAREFINE": f"{d['census_dense'][-1]['shared_frac']*100:.1f}",
        "SMALLSHAREFINE": f"{d['census'][-1]['shared_frac']*100:.1f}",
        "RUNROWS": rows,
        "NENTSMALL": f"{run('collide', lo)['n_entries']:,}",
        "NENTDENSE": f"{run('dense', lo)['n_entries']:,}",
        "LAMONEGAP": f"${run('collide', lo)['psnr_a'] - run('collide', lo)['psnr_b']:+.2f}$",
        "LAMONEGAPD": f"${run('dense', lo)['psnr_a'] - run('dense', lo)['psnr_b']:+.2f}$",
        "LAMONECOSTA": f"{cost('a', lo):.2f}", "LAMONECOSTB": f"{cost('b', lo):.2f}",
        "DENSEDRIFT": f"{run('dense', lo)['psnr_b'] - run('dense', hi)['psnr_b']:.2f}",
        "DENSEBONE": f"{run('dense', lo)['psnr_b']:.2f}",
        "DENSEBHUN": f"{run('dense', hi)['psnr_b']:.2f}",
        "DOMONE": f"{d['dominance'][f'collide_lam{lo:g}']['median']:.3f}",
        "DOMTEN": f"{d['dominance'][f'collide_lam{mid:g}']['median']:.3f}",
        "DOMHUN": f"{d['dominance'][f'collide_lam{hi:g}']['median']:.3f}",
        "DOMHUNFRAC": f"{d['dominance'][f'collide_lam{hi:g}']['frac_above_0.9']*100:.1f}",
        "NSHARED": f"{d['n_shared_hashed']:,}",
        "COSTAONE": f"{cost('a', lo):.2f}", "COSTATEN": f"{cost('a', mid):.2f}",
        "COSTAHUN": f"{cost('a', hi):.2f}",
        "COSTBONE": f"{cost('b', lo):.2f}", "COSTBTEN": f"{cost('b', mid):.2f}",
        "COSTBHUN": f"{cost('b', hi):.2f}",
        "COSTRATIO": f"{cost('b', hi)/max(1e-9, cost('a', hi)):.1f}",
        "PFONELOWN": f"{p1['0.0']['n_rows']:,}", "PFONEHIGHN": f"{p1['0.9']['n_rows']:,}",
        "PFONELOW": f"{share_a(p1['0.0'])*100:.0f}\\%",
        "PFONEHIGH": f"{share_a(p1['0.9'])*100:.0f}\\%",
        "PFHUNHIGHN": f"{ph['0.9']['n_rows']:,}",
        "PFHUNHIGHB": f"{ph['0.9']['drop_b']:.2f}",
    }

    src = open(os.path.join(HERE, "collision_audit.tex")).read()
    used = set()

    def sub(m):
        name = m.group(1)
        if name not in v:
            return m.group(0)                        # a real LaTeX macro
        used.add(name)
        return v[name]

    out = re.sub(r"\\([A-Z]{2,})(?:\{\})?", sub, src)
    missing = sorted(set(v) - used)
    left = sorted(set(re.findall(r"\\([A-Z]{2,})(?:\{\})?", out)) & set(v))
    if missing or left:
        sys.exit(f"placeholders unused: {missing}\nplaceholders unresolved: {left}")

    filled = os.path.join(HERE, "collision_audit.filled.tex")
    open(filled, "w").write(out)
    for _ in range(2):                               # twice, for the float refs
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "-jobname", "collision_audit",
                            os.path.basename(filled)],
                           cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed")
    for ext in (".aux", ".log", ".out", ".filled.tex"):
        f = os.path.join(HERE, "collision_audit" + ext)
        if os.path.exists(f):
            os.remove(f)
    print(f"wrote {os.path.join(HERE, 'collision_audit.pdf')} "
          f"({len(v)} numbers taken from {os.path.relpath(JSON, ROOT)})")


if __name__ == "__main__":
    main()
