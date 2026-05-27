# Pushing this to Hugging Face

Three commands to publish `samielakkad1/jakma-darija-classifier`.

## 1. Authenticate (one time)

Get a token from https://huggingface.co/settings/tokens (with `write` permission), then:

```bash
hf auth login
# paste your token
```

## 2. Create the repo

```bash
hf repo create samielakkad1/jakma-darija-classifier --type model --license apache-2.0
```

## 3. Push these files

From this folder (`/c/Users/SAMI EL AKKAD/repos/jakma-darija-classifier`):

```bash
hf upload samielakkad1/jakma-darija-classifier . --commit-message "Initial model card + handler + config"
```

Done. Model card will be live at https://huggingface.co/samielakkad1/jakma-darija-classifier

## Files in this repo

| File | What it is |
|---|---|
| `README.md` | The model card — methodology, dataset, eval slices, limitations |
| `config.json` | Architecture + label maps + LoRA config |
| `handler.py` | Inference handler for HF Spaces / Inference Endpoints |
| `requirements.txt` | Python dependencies |
| `PUSH_INSTRUCTIONS.md` | This file |

## What this does NOT include yet

- **`heads.pt`** — the LoRA-tuned classification head weights. To produce these, you need to:
  1. Export ~50K labeled queries from the jak.ma database (anonymized)
  2. Run LoRA fine-tuning on a T4 GPU (~$8 on RunPod, 2 hours)
  3. Save heads to `heads.pt` and re-upload via `hf upload`

A training script for step 2 is referenced in the model card (link to `jak-ma-eval-suite/scripts/finetune/`); not included here to keep this repo focused on the methodology + handler.

Until `heads.pt` is added, the handler will load **untrained random heads** — fine for showing the architecture publicly, not for production use. Mark the model card with a clear "weights pending — see roadmap" disclaimer if you publish before training.

## Why this is worth publishing even without trained heads

For recruiters checking your HF profile:
- A real model repo (not just a Space) with a serious model card
- Methodology documentation that's been peer-reviewed against Baidu ERNIE eval infrastructure
- Production-system provenance (live at jak.ma)
- Three linked open-source repos (pm-frameworks-darija, jak-ma-eval-suite, ernie-evaluation-notes)
- A clear architecture description with eval slices and honest limitations

This is the difference between "I built a HF Space" and "I have a published HF model repo with a methodology card." For R&D recruiters, the second matters significantly more.

## Privacy + ethics check before pushing

- [ ] PII removed from any dataset references (already done in the card text — no phone numbers / names / coordinates)
- [ ] Production metrics quoted are the ones you can defend (p50 1.18s, $0.0008/query, 0.7% verifier rejection — all from your case-study repo)
- [ ] License is correct (Apache 2.0 for code, CC-BY-4.0 for the card text — both permissive)
- [ ] Citation block has your name spelled correctly
- [ ] Affiliation is up to date (Tsinghua SIGS, MSc AI)
- [ ] Email is the Tsinghua one (sam25@mails.tsinghua.edu.cn), not personal Gmail
