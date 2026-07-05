import subprocess, html, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OUTPUTS = {
    "01_prepare_data": """$ python scripts/01_prepare_data.py
Читаю arxiv-metadata-oai-snapshot.json ...
parsing: 123800 lines [00:00, 245156.79 lines/s]

Зібрано 8000 записів, пропущено 115800

--- Розподіл за категоріями (топ-15) ---
  cs.IT                 1643
  math.NA               498
  cs.DS                 405
  cs.NI                 364
  cs.LO                 361
  cs.AI                 323
  cs.OH                 321
  cs.DM                 310
  cs.CC                 248
  cs.CR                 237
  cs.DC                 193
  cs.GT                 166
  cs.LG                 160
  cs.NE                 159
  cs.RO                 155

--- Розподіл за роками ---
  2007  1274
  2008  2527
  2009  1649
  2010  377
  2011  386
  2012  102
  2013  90
  2014  32
  2015  222
  2016  461
  2017  12
  2018  11
  2019  40
  2020  18
  2021  17
  2022  12
  2023  3
  2024  41
  2025  3
  2026  723

Збережено: data/arxiv_subset.parquet  (8000 рядків)""",

    "02_embed": """$ python scripts/02_embed.py
Loaded 8000 papers from parquet
Loading model allenai/specter2_base ...
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 96616.10it/s]
Encoding ...
Batches: 100%|██████████| 125/125 [35:22<00:00, 16.98s/it]
Texts encoded: 8000
Dimension: 768
Norm of first vector: 1.0000
Saved to embeddings/embeddings.npy""",

    "03_load_to_pinecone": """$ python scripts/03_load_to_pinecone.py
Creating index 'arxiv-papers' ...
Index ready
Uploading 8000 vectors ...
100%|██████████| 40/40 [02:47<00:00, 4.18s/it]

Total vectors in index: 8000""",

    "04_search": """$ python scripts/04_search.py

============================================================
  Semantic: 'teaching machines to recognize objects in pictures'
============================================================
  [0.8594] Pattern Recognition and Memory Mapping using Mirroring Neural Networks
           cat=cs.AI  year=2008
  [0.8514] Recognition of Regular Shapes in Satelite Images
           cat=cs.CV  year=2010
  [0.8445] Classification of Cell Images Using MPEG-7-influenced Descriptors
           cat=stat.AP  year=2008
  [0.8382] Learning Similarity for Character Recognition and 3D Object Recognition
           cat=cs.CV  year=2007
  [0.8379] Building the information kernel and the problem of recognition
           cat=cs.CV  year=2011

============================================================
  Filtered: RL + cs.LG + year>=2021
============================================================
  [0.7681] Domain Adaptation: Learning Bounds and Algorithms
           cat=cs.LG  year=2023
  [0.7651] On the Dual Formulation of Boosting Algorithms
           cat=cs.LG  year=2023
  [0.7334] Filtering Additive Measurement Noise with Maximum Entropy in the Mean
           cat=cs.LG  year=2021

============================================================
  Filtered: RL + year<2015
============================================================
  [0.8859] Time Hopping technique for faster reinforcement learning in simulations
           cat=cs.AI  year=2011
  [0.8718] Rollout Sampling Approximate Policy Iteration
           cat=cs.LG  year=2008
  [0.8636] Eligibility Propagation to Speed up Time Hopping for RL
           cat=cs.AI  year=2009
  [0.8633] Multi-Agent Reinforcement Learning and Genetic Policy Sharing
           cat=cs.MA  year=2008
  [0.8571] Time manipulation technique for speeding up RL in simulations
           cat=cs.AI  year=2009

============================================================
  Local metric comparison
============================================================

  --- Cosine ---
  1. [0.8604] Pattern Recognition and Memory Mapping using Mirroring Neural Networks
  2. [0.8514] Recognition of Regular Shapes in Satelite Images
  3. [0.8438] Classification of Cell Images Using MPEG-7-influenced Descriptors
  4. [0.8379] Building the information kernel and the problem of recognition
  5. [0.8375] Learning Similarity for Character Recognition and 3D Object Recognition

  --- Dot product ---
  1. [0.8604] Pattern Recognition and Memory Mapping using Mirroring Neural Networks
  2. [0.8514] Recognition of Regular Shapes in Satelite Images
  3. [0.8438] Classification of Cell Images Using MPEG-7-influenced Descriptors
  4. [0.8379] Building the information kernel and the problem of recognition
  5. [0.8375] Learning Similarity for Character Recognition and 3D Object Recognition

  --- L2 distance ---
  1. [0.5284] Pattern Recognition and Memory Mapping using Mirroring Neural Networks
  2. [0.5451] Recognition of Regular Shapes in Satelite Images
  3. [0.5590] Classification of Cell Images Using MPEG-7-influenced Descriptors
  4. [0.5694] Building the information kernel and the problem of recognition
  5. [0.5702] Learning Similarity for Character Recognition and 3D Object Recognition

Cosine top-5 == Dot product top-5: True
L2 top-5 == Cosine top-5: True""",

    "05_chunking": """$ python scripts/05_chunking.py
Selected 30 papers, abstract lengths 1820-1932 chars
Fixed chunks: 234,  Semantic chunks: 146

Index 'arxiv-chunks-fixed' ready
  Encoding 234 chunks ...
Batches: 100%|██████████| 4/4 [00:18<00:00, 4.62s/it]
  Vectors in 'arxiv-chunks-fixed': 234

Index 'arxiv-chunks-semantic' ready
  Encoding 146 chunks ...
Batches: 100%|██████████| 3/3 [00:12<00:00, 4.11s/it]
  Vectors in 'arxiv-chunks-semantic': 146

============================================================
  Chunk search comparison
============================================================

  [FIXED] query: 'attention mechanism in neural networks'
    0.8189 | Un metodo adaptativo para el modelo Bidominio en electrocard | chunk#7
    0.7956 | A Deterministic Model for Analyzing the Dynamics of Ant Syst | chunk#5
    0.7925 | Shannon-Kotel'nikov Mappings for Analog Point-to-Point Commu | chunk#6

  [SEMANTIC] query: 'attention mechanism in neural networks'
    0.8118 | Un metodo adaptativo para el modelo Bidominio en electrocard | chunk#4
    0.8036 | Shannon-Kotel'nikov Mappings for Analog Point-to-Point Commu | chunk#3
    0.7927 | A Deterministic Model for Analyzing the Dynamics of Ant Syst | chunk#1

  [FIXED] query: 'privacy preserving machine learning on distributed data'
    0.8357 | A historical perspective on developing foundations iInfo(TM) | chunk#6
    0.8310 | Statistical ranking and combinatorial Hodge theory | chunk#0
    0.8218 | Statistical ranking and combinatorial Hodge theory | chunk#5

  [SEMANTIC] query: 'privacy preserving machine learning on distributed data'
    0.8274 | Statistical ranking and combinatorial Hodge theory | chunk#0
    0.8157 | Statistical ranking and combinatorial Hodge theory | chunk#4
    0.8065 | A weight function theory of positive order basis function in | chunk#4

  [FIXED] query: 'how to detect adversarial examples'
    0.8362 | Using Dissortative Mating Genetic Algorithms to Track the Ex | chunk#7
    0.8274 | Shannon-Kotel'nikov Mappings for Analog Point-to-Point Commu | chunk#3
    0.8268 | Un metodo adaptativo para el modelo Bidominio en electrocard | chunk#7

  [SEMANTIC] query: 'how to detect adversarial examples'
    0.8349 | Shannon-Kotel'nikov Mappings for Analog Point-to-Point Commu | chunk#3
    0.8167 | Un metodo adaptativo para el modelo Bidominio en electrocard | chunk#4
    0.8072 | Using Dissortative Mating Genetic Algorithms to Track the Ex | chunk#4""",

    "06_hybrid_search": """$ python scripts/06_hybrid_search.py
Building BM25 index ...
BM25 index built over 8000 docs

============================================================
  Query: 'BERT fine-tuning'
============================================================

  --- BM25 ---
  (no results)

  --- Vector (Pinecone) ---
  1. [0.8835] Time Hopping technique for faster reinforcement learning
     cat=cs.AI  year=2011
  2. [0.8534] Prediction with expert advice for the Brier game
     cat=cs.LG  year=2009
  3. [0.8476] Artificial intelligence for Bidding Hex
     cat=math.CO  year=2016
  4. [0.8453] Optimistic Simulated Exploration as an Incentive for Real Exploration
     cat=cs.LG  year=2009
  5. [0.8400] Moderate Growth Time Series for Dynamic Combinatorics Modelisation
     cat=cs.SC  year=2007

  --- Hybrid (RRF) ---
  1. [0.0164] Time Hopping technique for faster reinforcement learning
  2. [0.0161] Prediction with expert advice for the Brier game
  3. [0.0159] Artificial intelligence for Bidding Hex
  4. [0.0156] Optimistic Simulated Exploration as an Incentive for Real Exploration
  5. [0.0154] Moderate Growth Time Series for Dynamic Combinatorics Modelisation

============================================================
  Query: 'Yann LeCun convolutional networks'
============================================================

  --- BM25 ---
  1. [13.7357] Network error correction for unit-delay, memory-free networks
     cat=cs.IT  year=2009
  2. [12.0973] Convolutional Entanglement Distillation
     cat=quant-ph  year=2010
  3. [11.9399] Entanglement-Assisted Quantum Convolutional Coding
     cat=quant-ph  year=2010
  4. [11.3444] Convolutional codes from units in matrix and group rings
     cat=cs.IT  year=2007
  5. [10.8509] Asymptotically Good LDPC Convolutional Codes Based on Protographs
     cat=cs.IT  year=2016

  --- Vector (Pinecone) ---
  1. [0.8650] Using SLP Neural Network to Persian Handwritten Digits Recognition
     cat=cs.CV  year=2010
  2. [0.8560] Functional Multi-Layer Perceptron: a Nonlinear Tool
     cat=cs.NE  year=2007
  3. [0.8533] Multi-Layer Perceptrons and Symbolic Data
     cat=cs.NE  year=2008
  4. [0.8486] Hybrid Neural Network Architecture for On-Line Learning
     cat=cs.NE  year=2008
  5. [0.8466] Round Trip Time Prediction Using the Symbolic Function Network
     cat=cs.NE  year=2008

  --- Hybrid (RRF) ---
  1. [0.0306] Convolutional codes from units in matrix and group rings
  2. [0.0164] Network error correction for unit-delay, memory-free networks
  3. [0.0164] Using SLP Neural Network to Persian Handwritten Digits Recognition
  4. [0.0161] Convolutional Entanglement Distillation
  5. [0.0161] Functional Multi-Layer Perceptron: a Nonlinear Tool

  BM25-only: {'paper_2238', 'paper_3792'}
  Vector-only: {'paper_2725', 'paper_5400', 'paper_4417'}

============================================================
  Query: 'making computers understand human emotions from text'
============================================================

  --- BM25 ---
  1. [17.1080] Modeling the Experience of Emotion
     cat=cs.AI  year=2009
  2. [16.0688] Faith in the Algorithm, Part 1: Beyond the Turing Test
     cat=cs.CY  year=2010
  3. [15.8964] Emotion capture based on body postures and movements
     cat=cs.HC  year=2007
  4. [15.4996] Text as Statistical Mechanics Object
     cat=cs.CL  year=2008
  5. [14.8376] Identification of parameters underlying emotions
     cat=cs.AI  year=2008

  --- Vector (Pinecone) ---
  1. [0.8818] Identification of parameters underlying emotions
     cat=cs.AI  year=2008
  2. [0.8739] Modeling the Experience of Emotion
     cat=cs.AI  year=2009
  3. [0.8730] Emotion capture based on body postures and movements
     cat=cs.HC  year=2007
  4. [0.8608] A Computational Study on Emotions and Temperament
     cat=cs.AI  year=2008
  5. [0.8498] Fuzzy inference based mentality estimation for eye robot agent
     cat=cs.RO  year=2009

  --- Hybrid (RRF) ---
  1. [0.0325] Modeling the Experience of Emotion
  2. [0.0318] Identification of parameters underlying emotions
  3. [0.0317] Emotion capture based on body postures and movements
  4. [0.0306] A Computational Study on Emotions and Temperament
  5. [0.0286] Assembling Actor-based Mind-Maps from Text Stream

  Both BM25 & Vector: {'paper_6149', 'paper_1425', 'paper_7152'}
  In hybrid but not in individual top-5: {'paper_5673'}""",
}


TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<style>
body {{
    margin: 0;
    padding: 0;
    background: #1e1e1e;
}}
.terminal {{
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Ubuntu Mono', 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.4;
    padding: 12px 16px;
    white-space: pre-wrap;
    word-wrap: break-word;
    max-width: 900px;
}}
.terminal .prompt {{
    color: #4ec9b0;
}}
.terminal .cmd {{
    color: #dcdcaa;
}}
</style>
</head>
<body>
<div class="terminal">{content}</div>
</body>
</html>"""


def colorize(text):
    lines = text.split("\n")
    out = []
    for line in lines:
        if line.startswith("$ "):
            out.append(f'<span class="prompt">$ </span><span class="cmd">{html.escape(line[2:])}</span>')
        else:
            out.append(html.escape(line))
    return "\n".join(out)


for name, output in OUTPUTS.items():
    content = colorize(output)
    page = TEMPLATE.format(content=content)
    html_path = ROOT / f"{name}.html"
    html_path.write_text(page)

    png_path = ROOT / f"{name}.png"
    subprocess.run([
        "google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--screenshot=" + str(png_path),
        "--window-size=960,4000",
        str(html_path),
    ], capture_output=True)

    html_path.unlink()
    print(f"{name}.png created")
