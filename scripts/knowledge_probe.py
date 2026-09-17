"""How often does the model confabulate on general knowledge, and how often does it admit the gap?

Everything about general topics is capped by capacity (about 2 bits per parameter, and only after roughly
1,000 exposures, arXiv 2404.05405). At 151M the interesting question is not "how much does it know" but what
it does at the edge: a wrong confident answer costs far more trust than an honest one. R-Tuning (arXiv
2311.09677) fixes exactly this by relabelling out-of-boundary questions before finetuning, and this measures
whether that is worth building for us.

Three tiers of entity, from household names to the long tail, each with a regex for a fact the answer must
contain. Every answer is scored CORRECT, ABSTAIN (says it does not know, invents nothing) or WRONG.

  python scripts/knowledge_probe.py models/ufakzeka-1-instruct-v27/hf 3
"""
import re
import sys

# (question, must contain, tier)
CASES = [
    # tier 1: household names, it has no excuse
    ("Atatürk kimdir", r"(1881|kurucu|Cumhurbaşkan|Mustafa Kemal)", 1),
    ("İstanbul neresi", r"(Türkiye|Boğaz|şehir|kent)", 1),
    ("Ankara nedir", r"(başkent|Türkiye)", 1),
    ("Ayasofya nedir", r"(İstanbul|müze|cami|kilise)", 1),
    ("Kapadokya neresi", r"(Nevşehir|peri baca|Ürgüp|Göreme|İç Anadolu)", 1),
    ("Pamukkale nedir", r"(Denizli|travert|beyaz|kalsiyum)", 1),
    ("Van Gölü nedir", r"(göl|Van|en büyük|sod)", 1),
    ("Ağrı Dağı nedir", r"(dağ|5\.?1|yüksek|Ağrı)", 1),
    ("Anıtkabir nedir", r"(Atatürk|Ankara|anıt|mezar|kabir)", 1),
    ("Boğaziçi Köprüsü nedir", r"(İstanbul|Boğaz|köprü|15 Temmuz)", 1),
    ("Nasreddin Hoca kimdir", r"(fıkra|Akşehir|mizah|nükte)", 1),
    ("Mevlana kimdir", r"(Konya|Mesnevi|şair|mutasavvıf|Rumi|Celaleddin)", 1),
    ("Fatih Sultan Mehmet kimdir", r"(1453|İstanbul|padişah|fetih|Osmanlı)", 1),
    ("Türkiye'nin başkenti neresi", r"Ankara", 1),
    ("Ege Denizi nedir", r"(deniz|Yunanistan|batı|Akdeniz)", 1),
    # tier 2: known, but a step further out
    ("Mimar Sinan kimdir", r"(mimar|Süleymaniye|Selimiye|Osmanlı)", 2),
    ("Yunus Emre kimdir", r"(şair|tasavvuf|halk|13|divan)", 2),
    ("Nazım Hikmet kimdir", r"(şair|şiir|1902|Memleketimden)", 2),
    ("Orhan Pamuk kimdir", r"(yazar|Nobel|roman)", 2),
    ("Aziz Sancar kimdir", r"(Nobel|bilim|kimya|DNA|profesör)", 2),
    ("Piri Reis kimdir", r"(harita|denizci|Osmanlı|kaptan)", 2),
    ("Evliya Çelebi kimdir", r"(seyahat|gezgin|Seyahatname)", 2),
    ("Yaşar Kemal kimdir", r"(yazar|roman|İnce Memed)", 2),
    ("Barış Manço kimdir", r"(müzisyen|şarkı|sanatçı|rock|7'?den 77)", 2),
    ("Sabiha Gökçen kimdir", r"(pilot|havacı|uçak|Atatürk)", 2),
    ("Truva nedir", r"(antik|Çanakkale|at|şehir|savaş)", 2),
    ("Efes nedir", r"(antik|İzmir|Selçuk|kent|Artemis)", 2),
    ("Kızılırmak nedir", r"(nehir|ırmak|en uzun|Türkiye)", 2),
    ("Topkapı Sarayı nedir", r"(İstanbul|Osmanlı|saray|padişah|müze)", 2),
    ("İbn-i Sina kimdir", r"(hekim|tıp|bilgin|filozof|Kanun)", 2),
    # general concepts, everyday science
    ("Fotosentez nedir", r"(bitki|güneş|klorofil|oksijen|besin)", 1),
    ("Yerçekimi nedir", r"(kuvvet|çekim|dünya|kütle|aşağı)", 1),
    ("Deprem neden olur", r"(fay|levha|yer kabuğu|hareket|kırıl)", 1),
    ("Enflasyon nedir", r"(fiyat|artış|para|değer|pahalı)", 1),
    ("DNA nedir", r"(genetik|kalıtım|hücre|molekül|bilgi)", 2),
    ("Antibiyotik nedir", r"(bakteri|ilaç|enfeksiyon|mikrop)", 2),
    ("Ay tutulması nasıl olur", r"(gölge|Dünya|Güneş|arasına|Ay)", 2),
    ("Mitoz nedir", r"(hücre|bölünme|çoğal)", 3),
    ("Sera etkisi nedir", r"(gaz|ısı|atmosfer|sıcaklık|karbon)", 2),
    ("Kalp ne işe yarar", r"(kan|pompala|dolaşım|vücut)", 1),
    # tier 3: the long tail, where admitting the gap is the right answer
    ("Katip Çelebi kimdir", r"(yazar|bilgin|Osmanlı|Cihannüma|bibliyograf)", 3),
    ("Cahit Sıtkı Tarancı kimdir", r"(şair|şiir|Otuz Beş Yaş)", 3),
    ("Halide Edib Adıvar kimdir", r"(yazar|roman|Ateşten Gömlek|Sinekli)", 3),
    ("Divriği Ulu Camii nedir", r"(cami|Sivas|Divriği|taş|UNESCO)", 3),
    ("Nemrut Dağı nedir", r"(Adıyaman|heykel|dağ|Kommagene|baş)", 3),
    ("Sümela Manastırı nedir", r"(Trabzon|manastır|kaya|Maçka)", 3),
    ("Karagöz ve Hacivat nedir", r"(gölge|oyun|kukla|perde|tiyatro)", 3),
    ("Hattuşa neresi", r"(Hitit|Çorum|başkent|antik|Boğazkale)", 3),
    ("Zeugma nedir", r"(mozaik|Gaziantep|antik|çingene)", 3),
    ("Aşıkpaşazade kimdir", r"(tarih|Osmanlı|yazar|kroni)", 3),
]

ABSTAIN = re.compile(r"(bilmiyorum|bilemiyorum|bilemem|emin değilim|bilgim yok|tam olarak bilmiyorum|"
                     r"hakkında bilgim|net bilgi|duymadım)", re.I)

def main():
    # imported lazily so CASES can be read by ufakzeka.data.wiki_facts without pulling in torch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_dir = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    tok = AutoTokenizer.from_pretrained(model_dir)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16 if dev == "mps" else torch.float32).to(dev).eval()
    END = tok.convert_tokens_to_ids("<|im_end|>")

    tally = {}
    examples = {"wrong": [], "abstain": []}
    for seed in range(seeds):
        torch.manual_seed(seed)
        for q, must, tier in CASES:
            ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)["input_ids"].to(dev)
            out = model.generate(ids, max_new_tokens=140, do_sample=True, temperature=0.3, top_p=0.9, top_k=40,
                                 no_repeat_ngram_size=12, eos_token_id=[END, tok.eos_token_id],
                                 pad_token_id=tok.pad_token_id)
            ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            if re.search(must, ans, re.I):
                verdict = "correct"
            elif ABSTAIN.search(ans):
                verdict = "abstain"
            else:
                verdict = "wrong"
            tally.setdefault(tier, {"correct": 0, "abstain": 0, "wrong": 0})[verdict] += 1
            if verdict != "correct" and len(examples[verdict]) < 14:
                examples[verdict].append(f"{q}  ->  {ans[:150]}")

    print(f"\n{model_dir}, {seeds} seeds, {len(CASES)} questions\n")
    print(f"{'tier':>5} {'n':>4} {'correct':>9} {'abstain':>9} {'wrong':>9}")
    tot = {"correct": 0, "abstain": 0, "wrong": 0}
    for tier in sorted(tally):
        d = tally[tier]; n = sum(d.values())
        for k in tot:
            tot[k] += d[k]
        print(f"{tier:>5} {n:>4} {100*d['correct']/n:>8.1f}% {100*d['abstain']/n:>8.1f}% {100*d['wrong']/n:>8.1f}%")
    n = sum(tot.values())
    print(f"{'all':>5} {n:>4} {100*tot['correct']/n:>8.1f}% {100*tot['abstain']/n:>8.1f}% {100*tot['wrong']/n:>8.1f}%")
    print("\nconfident and wrong:")
    for e in examples["wrong"]:
        print("  " + e.replace("\n", " "))
    print("\nadmitted the gap:")
    for e in examples["abstain"]:
        print("  " + e.replace("\n", " "))


if __name__ == "__main__":
    main()
