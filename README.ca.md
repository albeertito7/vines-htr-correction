# Fine-tuning de LLMs per a post-correcció d'HTR

## Els diaris manuscrits de Ramon Viñes (1890–1915)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![Model on HuggingFace](https://img.shields.io/badge/🤗_Model-HuggingFace-yellow.svg)](https://huggingface.co/albertdatzira/qwen3-8b-htr-vines-lora)

> 📖 **[English README](README.md)**

Pipeline complet per a la **post-correcció de text reconegut per HTR (Handwritten Text Recognition) mitjançant models de llenguatge (LLM) amb fine-tuning**, aplicat als diaris manuscrits del pianista català Ramon Viñes (1890–1915).

### Resultat principal

Un LLM d'ús general (Llama 3.1 70B) **empitjora** la sortida HTR un 95% per sobrecorrecció de l'ortografia original de l'autor. Un model fine-tuned (Qwen3-8B, 9× més petit) la **millora** un 28,2%, demostrant que el fine-tuning específic és essencial.

### Taula de resultats

| Enfocament | CER | Pàg. empitjorades | vs. Baseline |
|-----------|------|-------------------|--------------|
| HTR sol (Balakirev) | 10,86% | — | — |
| Llama 3.1 70B (general, sense fine-tuning) | 21,20% | totes | +95% (pitjor) |
| Qwen3-8B fine-tuned (LoRA) | 8,72% | 2 | −19,7% |
| **Qwen3-8B fine-tuned + fallback** | **7,80%** | **0** | **−28,2%** |

*Avaluat sobre 18 pàgines de test mai vistes (RV_1908, RV_1915). Veure [Avaluació sobre el corpus complet](#avaluació-sobre-el-corpus-complet) per als resultats globals.*

---

## Índex

- [Resum](#resum)
- [Com hem usat el pipeline](#com-hem-usat-el-pipeline)
- [Estructura del repositori](#estructura-del-repositori)
- [Requisits previs](#requisits-previs)
- [Inici ràpid](#inici-ràpid)
- [Visió general del pipeline](#visió-general-del-pipeline)
- [Ús](#ús)
- [Configuració](#configuració)
- [Re-entrenar amb noves dades](#re-entrenar-amb-noves-dades)
- [Avaluació sobre el corpus complet](#avaluació-sobre-el-corpus-complet)
- [Model pre-entrenat](#model-pre-entrenat)
- [Imatges Docker](#imatges-docker)
- [Notes DGX Spark](#notes-dgx-spark)
- [Citació](#citació)
- [Agraïments](#agraïments)
- [Llicència](#llicència)

---

## Resum

Els diaris de Ramon Viñes (1890–1943) són una font primària única per a la musicologia i la història cultural. El grup de recerca GReia de la Universitat de Lleida els digitalitza amb eScriptorium i un model Kraken fine-tuned (Balakirev, basat en FoNDUE-GD v3).

Balakirev aconsegueix un CER global del **17,54%** sobre 659 pàgines, amb variació entre anys (9,12%–29,95%). Aquest pipeline aplica post-correcció amb LLM per reduir els errors preservant l'ortografia irregular de l'autor ("brila", "dibajo", "murchachos" són formes intencionals, no errors d'HTR).

El repte principal és distingir errors d'HTR de l'escriptura original — un problema que els LLMs generals no poden resoldre sense fine-tuning específic.

---

## Com hem usat el pipeline

Aquesta secció descriu els passos exactes que hem seguit, per a la reproducibilitat.

### 1. Generat la sortida HTR crua

S'ha executat el model Balakirev sobre les 659 imatges de pàgina del repositori [RV-training](https://github.com/esthersole/RV-training) per generar la sortida HTR crua (ALTO XML amb errors). ~1,5 hores a la DGX Spark.

### 2. Mesurat la baseline

S'ha comparat la sortida HTR crua contra el ground truth validat per humans. CER global: **17,54%**, WER: **36,73%** sobre 659 pàgines (17 anys).

### 3. Provat un LLM general

S'ha aplicat Llama 3.1 70B (via Ollama) sense fine-tuning. El CER ha **empitjorat** de 10,63% a 21,20% (+99,5%) perquè el model "corregia" l'ortografia original de l'autor ("brila"→"brilla", "murchachos"→"muchachos"). Això ha confirmat que el fine-tuning és essencial.

### 4. Preparat les dades d'entrenament

S'han creat 626 parells d'entrenament a nivell de pàgina (text HTR amb errors → text GT correcte), filtrant portades (GT < 50 chars) i pàgines amb errors estructurals (CER > 80%). Les dades s'han dividit **per any** (no aleatòriament) per evitar data leakage:
- **Train**: 583 pàgines de 13 anys
- **Validació**: 25 pàgines de 2 anys (RV_1892-1893, RV_1909)
- **Test**: 18 pàgines de 2 anys (RV_1908, RV_1915)

### 5. Fine-tuning de Qwen3-8B amb LoRA

S'ha entrenat Qwen3-8B amb LoRA (rank 16, alpha 32). L'entrenament ha trigat **56 minuts** a la DGX Spark, entrenant 43,6M de paràmetres (0,53% del model).

### 6. Avaluat i iterat

El model fine-tuned ha reduït el CER de 10,86% a **7,80%** sobre el test set (28,2% de millora) amb un mecanisme de fallback que garanteix **zero pàgines empitjorades**.

També hem descobert que:
- **La inferència per blocs falla**: el model entrenat amb pàgines senceres no generalitza a blocs curts (distribution shift).
- **El rendiment varia al corpus complet**: el model funciona millor en pàgines amb CER baix (<12%) i menys en pàgines difícils.
- **El fallback és essencial**: sense ell, el truncament i les al·lucinacions empitjoren 2 de 18 pàgines.

### 7. Aplicat al corpus complet

S'ha executat el model sobre les 713 pàgines amb checkpointing i doble fallback (truncament + comparació de CER contra GT).

---

## Estructura del repositori

```
vines-htr-correction/
├── Dockerfile                     # Imatge del pipeline (parser, mètriques)
├── Dockerfile.kraken              # Imatge Kraken + Balakirev
├── Dockerfile.finetune            # Imatge fine-tuning (NVIDIA NGC base)
├── docker-compose.yml             # Orquestració de serveis
├── config.yaml                    # Configuració global del pipeline
├── requirements.txt               # Dependències Python (pipeline)
├── requirements-finetune.txt      # Dependències Python (fine-tuning)
├── LICENSE                        # Llicència MIT
├── README.md                      # README en anglès
├── README.ca.md                   # Aquest fitxer
│
├── configs/
│   └── finetune_config.yaml       # Model, LoRA i hiperparàmetres
│
├── results/                       # Resultats de referència (tracked)
│   ├── README.md                  # Descripció de cada fitxer
│   ├── baseline_cer.json          # CER/WER global del Balakirev
│   ├── baseline_summary.csv       # CER/WER per any
│   ├── training_config.json       # Configuració del fine-tuning
│   ├── finetuned_test_summary.json     # Resum test set (CER 7,80%)
│   ├── finetuned_test_detail.json      # Resultats per pàgina
│   ├── finetuned_easy_summary.json     # Pàgines fàcils
│   └── finetuned_moderate_summary.json # Pàgines moderades
│
├── src/                           # Codi font del pipeline
│   ├── parsers/                   # Parser ALTO XML
│   ├── preprocessing/             # Retallador d'imatges de línia
│   ├── correction/                # Correctors (cloud, local, prompts)
│   ├── evaluation/                # Mètriques CER/WER
│   ├── pipeline/                  # Benchmark, preparació dades, producció
│   └── export/                    # HTML diffs, upload eScriptorium
│
├── scripts/                       # Scripts d'execució
│   ├── legacy/                    # Scripts de la fase inicial
│   └── docker/                    # Scripts Docker wrapper
│
├── data/.gitkeep                  # Dades generades (no tracked)
├── models/.gitkeep                # Models fine-tuned (no tracked)
├── benchmark/.gitkeep             # Avaluacions (no tracked)
└── logs/.gitkeep                  # Logs (no tracked)
```

---

## Requisits previs

- **Docker** (amb suport GPU per fine-tuning i inferència)
- **Dades de ground truth**: el repositori [RV-training](https://github.com/esthersole/RV-training) (privat, demanar accés a l'equip GReia)
- **GPU** (opcional): NVIDIA GPU per fine-tuning. CPU funciona per parsing i avaluació.

---

## Inici ràpid

```bash
# 1. Clonar el repositori
git clone https://github.com/albeertito7/vines-htr-correction.git
cd vines-htr-correction

# 2. Clonar el repositori de ground truth (requereix accés)
git clone https://github.com/esthersole/RV-training.git ~/RV-training

# 3. Construir les imatges Docker
bash scripts/docker/build.sh

# 4. Generar la sortida HTR crua (~1,5 hores)
bash scripts/docker/run_balakirev_all.sh

# 5. Calcular el CER baseline
bash scripts/docker/run_baseline.sh

# 6. Preparar les dades de fine-tuning
bash scripts/docker/run_prepare_finetuning.sh

# 7. Fine-tuning (~56 minuts a la DGX Spark)
bash scripts/docker/run_finetune.sh

# 8. Avaluar
bash scripts/docker/run_evaluate.sh test
```

---

## Visió general del pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                 Imatges de pàgina (PNG/JPG)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Kraken / Balakirev (motor HTR)                  │
│       Segmentació + OCR → ALTO XML (amb errors)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Parser ALTO XML                           │
│           Extreure text + coordenades de línia               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            LLM Fine-tuned (Qwen3-8B + LoRA)                 │
│    Correcció de pàgina completa amb mecanisme de fallback    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Avaluació / Exportació                      │
│       Mètriques CER/WER, HTML diffs, eScriptorium           │
└─────────────────────────────────────────────────────────────┘
```

---

## Ús

### Generar la sortida HTR crua

```bash
bash scripts/docker/run_balakirev_all.sh              # Tot el corpus (~1,5h)
bash scripts/docker/run_balakirev_all.sh RV_1906      # Només un any
bash scripts/docker/run_balakirev_page.sh RV_1890 page_00005  # Una pàgina
```

### Calcular el CER baseline

```bash
bash scripts/docker/run_baseline.sh
```

### Preparar dades i entrenar

```bash
bash scripts/docker/run_prepare_finetuning.sh         # Preparar parells
bash scripts/docker/run_analyze_outliers.sh           # Analitzar outliers
bash scripts/docker/run_finetune.sh                   # Entrenar (~56 min)
```

### Avaluar

```bash
bash scripts/docker/run_evaluate.sh                   # Test set (18 pàgines)
bash scripts/docker/run_evaluate.sh train 10          # Verificar amb 10 del train
bash scripts/docker/run_production_cer.sh             # CER de producció
```

### Producció

```bash
nohup bash scripts/docker/run_production.sh > logs/production.log 2>&1 &
tail -f logs/production.log
```

Inclou checkpointing — es pot interrompre i reprendre.

---

## Configuració

Tots els paràmetres es troben a `configs/finetune_config.yaml`. Per canviar de model, una sola línia:

```yaml
model:
  name: "Qwen/Qwen3-8B"                               # Per defecte
  # name: "meta-llama/Llama-3.1-8B-Instruct"           # Llama
  # name: "mistralai/Mistral-7B-Instruct-v0.3"         # Mistral
```

Veure [`configs/finetune_config.yaml`](configs/finetune_config.yaml) per a tots els paràmetres disponibles.

---

## Re-entrenar amb noves dades

Quan hi hagi noves pàgines de ground truth:

```bash
# 1. Afegir els nous XML + imatges al repositori RV-training
# 2. Executar Balakirev (salta les ja processades)
bash scripts/docker/run_balakirev_all.sh
# 3. Regenerar dades d'entrenament
bash scripts/docker/run_prepare_finetuning.sh
# 4. Entrenar (~56 minuts)
bash scripts/docker/run_finetune.sh
# 5. Avaluar
bash scripts/docker/run_evaluate.sh test
```

Tot el cicle triga aproximadament 1 hora.

---

## Avaluació sobre el corpus complet

Els resultats del test set (CER 7,80%, 89% millorades) són forts, però l'avaluació sobre tot el corpus mostra resultats més matisats:

| Escenari | Pàgines | CER abans | CER després | Millorades |
|----------|---------|-----------|-------------|------------|
| Test set (RV_1908, RV_1915) | 18 | 10,86% | 7,80% | 16 (89%) |
| Pàgines fàcils, altres anys (CER <12%) | 10 | 10,37% | 9,66% | 4 (40%) |
| Pàgines moderades (CER 14-19%) | 10 | 17,06% | 16,81% | 2 (20%) |
| Pàgines difícils (CER >30%) | — | — | — | 0% (fallback) |

**En tots els escenaris, zero pàgines empitjoren** gràcies al mecanisme de fallback.

El factor limitant és la diversitat de dades d'entrenament (583 exemples de 13 anys). Amb dades addicionals i data augmentation, s'espera millorar la proporció de pàgines corregides.

---

## Model pre-entrenat

Els adaptadors LoRA fine-tuned estan disponibles a HuggingFace:

**[albertdatzira/qwen3-8b-htr-vines-lora](https://huggingface.co/albertdatzira/qwen3-8b-htr-vines-lora)**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
model = PeftModel.from_pretrained(base, "albertdatzira/qwen3-8b-htr-vines-lora")
```

Veure el [model card](https://huggingface.co/albertdatzira/qwen3-8b-htr-vines-lora) per a instruccions completes, limitacions i configuració de fallback recomanada.

---

## Imatges Docker

| Imatge | Base | Mida | Funció |
|--------|------|------|--------|
| `vines-pipeline` | `python:3.12-slim` | ~350 MB | Parsing, avaluació, preparació de dades |
| `vines-kraken` | `python:3.12-slim` + PyTorch | ~1,7 GB | Executar Balakirev sobre les imatges |
| `vines-finetune` | `nvcr.io/nvidia/pytorch:25.11-py3` | ~20 GB | Fine-tuning i inferència (GPU) |

---

## Notes DGX Spark

| Component | Especificació |
|-----------|--------------|
| GPU | NVIDIA GB10 (Blackwell, SM121) |
| CPU | 20 cores ARM (Cortex-X925 + Cortex-A725) |
| Memòria | 128 GB unificada (compartida CPU/GPU) |
| CUDA | 13.0 |
| OS | Ubuntu 24.04 |

Consideracions:
- **ARM64**: usar la imatge NGC com a base Docker.
- **CUDA 13**: paquets compilats per CUDA 12.x no funcionen.
- **Memòria unificada**: un OOM pot congelar tot el sistema.
- **Sense sudo**: tot s'executa dins Docker.
- **torchao**: desinstal·lar de la imatge NGC (incompatible amb PEFT).

---

## Citació

```bibtex
@mastersthesis{albert2026finetuning,
    title={Fine-tuning Large Language Models for HTR Post-correction:
           A Case Study on the Ramon Viñes Manuscript Diaries},
    author={Pérez Datsira, Albert},
    school={Universitat de Lleida},
    year={2026},
    type={Treball de Fi de Màster}
}
```

---

## Agraïments

- **Carles Mateu** — director del treball i accés a la infraestructura DGX Spark
- **Cèsar Fernández Camón** — co-director del treball
- **Esther Solé i Martí** — historiadora de l'art, gestió del ground truth i eScriptorium
- **Màrius Josep Bernadó Tarragona** — musicòleg, coordinació del projecte dins les humanitats digitals
- [**Grup de recerca GREiA**](https://greia.udl.cat/) — Universitat de Lleida, col·laboració i suport

---

## Llicència

Aquest projecte està llicenciat sota la [MIT License](LICENSE).

El model base Qwen3-8B està llicenciat sota [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Les imatges i transcripcions dels diaris de Ramon Viñes són gestionades pel grup GReia i no estan incloses en aquest repositori.
