# mRNA Concept Datasets

Harvest commands ready to copy-paste.
Status: ✅ done  🔁 needs re-harvest  🔜 queued

Key lesson: training data must match the prompt STYLE the router will see at inference.
Raw text datasets (case excerpts, poems, abstracts) cause style mismatch when
the router receives conversational/instruction inputs. All datasets below are
instruction or Q&A format.

--- SCIENCE TRIAD

## ✅ Biology (first real .mrna adapter + SAE concept)
20k chat Q&A pairs. Distinctive domain. Used for both adapter training and SAE harvesting.

**Harvest activations for SAE:**
```bash
python sandbox-scripts/harvest_hf.py \
    --dataset camel-ai/biology \
    --text-column message_1 \
    --concept biology \
    --max-examples 5000 \
```
Then add `biology:data/biology_layer*.pt` to the `train_sae.py` command and retrain.
openlifescienceai/medmcqa question, exp cols
**Train the adapter:**
```bash
python sandbox-scripts/train_mrna_adapter.py \
    --dataset camel-ai/biology \
    --concept biology \
```
Saves to `adapters/biology_lora/` in PEFT format — vLLM `LoRARequest` loads it directly.

## ✅ Physics (next adapter + SAE concept)
```bash
# Harvest
python sandbox-scripts/harvest_hf.py \
    --dataset camel-ai/physics \
    --text-column message_1 --concept physics \
    --max-examples 5000

# Train adapter
python sandbox-scripts/train_mrna_adapter.py \
    --dataset camel-ai/physics \
    --concept physics
```

## ✅ Chemistry
```bash
python sandbox-scripts/harvest_hf.py \
    --dataset camel-ai/chemistry \
    --text-column message_1 --concept chemistry \
    --max-examples 5000

python sandbox-scripts/train_mrna_adapter.py \
    --dataset camel-ai/chemistry \
    --concept chemistry \
    --output-dir data/gemma-4-e2b/adapters \
    --max-examples 5000
```
camel-ai series. Same format as biology/physics.

---

## 🔁 Python
```bash
python sandbox-scripts/harvest_hf.py \
    --dataset iamtarun/python_code_instructions_18k_alpaca \
    --text-column instruction \
    --concept python

python sandbox-scripts/train_mrna_adapter.py \
    --dataset iamtarun/python_code_instructions_18k_alpaca \
    --split train \
    --text-column instruction \
    --concept python \
    --output-dir data/gemma-4-e2b/adapters \
    --max-examples 5000
```
Alpaca format. Style matches real coding prompts. Working well.

---
cais/hle

## 🔁 Legal

---

## 🔁 Medical
FreedomIntelligence/Medical-O1-Verifiable-Problem-Solving
---

## 🔁 Reasoning
Magpie-Align/Magpie-Gemma-2-9B-Instruct-Chat-v0.1
---

## 🔁 Astrophysics
Tijmen2/cosmosage-v3.1 or NASA-Chandra/astronomy-vqa
```bash
python sandbox-scripts/harvest_hf.py \
    --dataset Tijmen2/cosmosage-v3.1 \
    --text-column instruction \
    --concept python \
```
---

## 🔁 Finance

---

## 🔁 Poetry
merve/poetry author, content, poem name, age, type cols

---

## 🔁 History

---

## 🔁 Conversation

---

## 🔁 Social / Preference (optional)

---

## Future: Ojibwe
Niigaane corpus (To-Lead/) — ~2000+ pairs, pending annotation and curation.
Will need instruction-format pairs, not raw text, for the same style-matching reason.
Slot name: `ojibwe`

---
## 🔁 Math - probably don't, base model + science triad negate by it
candidates: HuggingFaceH4/MATH-500 test split, problem, solution, answer cols
meta-math/MetaMathQA train split, query, original_question, response cols
```bash
# New Math-500 Harvest command (test-only split)
python sandbox-scripts/harvest_hf.py \
    --dataset HuggingFaceH4/MATH-500 \
    --split test \
    --text-column problem \
    --concept math \
    --max-examples 500

python sandbox-scripts/train_mrna_adapter.py \
    --concept math \
    --split test \
    --dataset HuggingFaceH4/MATH-500 \
    --text-column problem
```
---

## Train once re-harvests are done
```bash
python sandbox-scripts/train_sae.py \
    --activations biology:data/gemma-4-e2b/activations/layer_25/biology_train.pt \
    chemistry:data/gemma-4-e2b/activations/layer_25/chemistry_train.pt \
    math:data/gemma-4-e2b/activations/layer_25/math_train.pt \
    physics:data/gemma-4-e2b/activations/layer_25/physics_train.pt \
    python:data/gemma-4-e2b/activations/layer_25/python_train.pt

# Benchmark the F1/Accuracy bleed matrix using our harvested holdout splits
python sandbox-scripts/eval_sae.py \
    --model-id gemma-4-e2b \
    --layer 25 \
    --concepts biology,chemistry,physics,math,python
```
