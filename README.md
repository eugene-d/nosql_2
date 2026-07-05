# Завдання 2 -- Семантичний пошук за науковими статтями

Дані -- arXiv JSONL дамп, фільтрація CS-статей, specter2 ембеддинги, завантаження в Pinecone, пошук різними способами.

## Структура

```
scripts/
  01_prepare_data.py   - фільтрація CS-статей, parquet
  02_embed.py          - specter2_base ембеддинги
  03_load_to_pinecone.py - завантаження в індекс
  04_search.py         - семантичний + фільтрований + порівняння метрик
  05_chunking.py       - fixed/semantic chunking
  06_hybrid_search.py  - BM25 + vector + RRF
```

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make all
```

Скрипти запускаються послідовно -- кожен залежить від попереднього.

Скріншоти виводу кожного скрипту -- в папці `screenshots/`.

---

## Частина 1 -- Підготовка даних і вибір інструментів

### 1.1 Дані

Скрипт `01_prepare_data.py` читає arxiv JSONL (~2.4M статей), фільтрує CS-категорії, бере перші 8000 і зберігає в parquet. CS обрано тому що там найбільше ML/NLP статей і семантичний пошук по них має сенс -- фізики і математики мають іншу лексику, і specter2 на них працює гірше.

### 1.2 Pinecone vs Qdrant vs Chroma

Pinecone -- повністю managed, нічого хостити не треба. Зареєструвався, отримав API key, кидаєш вектори через HTTP. Це досить зручно коли налаштування Docker/infra для завдань подібних до такого проєкту достатньо, щоб виконати роботу. Із обмежень безкоштовного тіру -- обмежена кількість індексів та 100K векторів максимум.

Qdrant -- open-source, Apache 2.0. Можна підняти локально або в їхньому cloud. Для серйозного проекту він має переваги такі як повний контроль, дані нікуди не їдуть, і на мільйонах векторів можна шардувати як треба. Для проєктів подібних домашній роботі це більше overkill.

Chroma дуже зручний інструмент через свої особливості запуску. Вона in-process, запускається прямо з Python без сервера. Для прототипів і ноутбуків це може бути один із кращих інструментів.

### 1.3 Чому specter2_base

Для пошуку по наукових текстах потрібна модель, яка розуміє академічну мову. all-MiniLM-L6-v2 -- хороша загальна модель, але вона навчена на generic текстах (Wikipedia, news, forums). specter2_base навчена конкретно на наукових публікаціях -- citation graphs, paper titles, abstracts з Semantic Scholar. На картці моделі на HuggingFace написано:

> "SPECTER 2.0 can be used to generate embeddings for scientific documents -- for tasks including classification, regression, retrieval, and search."

Тобто модель цілеспрямовано тренували на retrieval наукових текстів. На датасеті arxiv specter2 дає відчутно кращий recall, ніж MiniLM -- при порівнянні на кількох запитах specter2 знаходить релевантніші статті, особливо коли запит сформульований не тими словами що в анотації.

### Метрика схожості

В картці specter2 рекомендується cosine similarity. Це важливо при створенні індексу -- якщо створити індекс з metric="euclidean", а модель повертає вектори які мають сенс тільки за cosine, то ранжування буде неправильне. В цьому проєкті індекс створено з metric="cosine".

### Cosine vs Dot Product для нормалізованих ембеддингів

Коли ембеддинги нормалізовані (||v|| = 1), cosine similarity стає тотожною dot product. Формула cosine sim:

```
cos(a, b) = (a . b) / (||a|| * ||b||)
```

Якщо ||a|| = ||b|| = 1, то знаменник = 1, і залишається просто a . b -- тобто dot product. Тому в 02_embed.py використовується normalize_embeddings=True -- це дозволяє застосовувати швидший dot product замість cosine без втрати якості.

---

## Частина 2 -- Завантаження в Pinecone

Скрипт `03_load_to_pinecone.py` створює serverless індекс і заливає вектори батчами по 200. В метаданих abstract обрізається до 500 символів бо Pinecone має ліміт 40KB на вектор. Повний текст залишається в parquet.

### Вивід скрипту

```
Creating index 'arxiv-papers' ...
Index ready
Uploading 8000 vectors ...
100%|██████████| 40/40 [02:47<00:00, 4.18s/it]

Total vectors in index: 8000
```

---

## Частина 3 -- Пошукові запити

### Семантичний пошук

Запит "teaching machines to recognize objects in pictures" знаходить статті про computer vision, object detection, image classification -- навіть якщо в анотації немає слова "pictures". Це і є суть семантичного пошуку.

### Порівняння метрик: cosine, dot product, L2

Топ-5 для cosine і dot product виявились ідентичними -- і це очікувано, бо ембеддинги нормалізовані (норма першого вектора ~1.0000). Для нормалізованих векторів cos(a,b) = a.b, тому ранжування однакове.

L2 (евклідова відстань) дає той самий порядок. Для одиничних векторів:

```
||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b) = 2 - 2*cos(a,b)
```

Тобто L2 -- монотонна трансформація cosine: менша відстань = більша схожість. Порядок зберігається.

Без нормалізації все б зламалось: dot product починає фаворизувати довші вектори (більша норма = більший скалярний добуток), а L2 отримує зміщення від різниці в нормах. Cosine єдина метрика яка коректно працює з ненормалізованими векторами, бо вона ділить на норми.

### Вивід скрипту

```
Semantic: 'teaching machines to recognize objects in pictures'
  [0.8594] Pattern Recognition and Memory Mapping using Mirroring Neural Networks (cs.AI, 2008)
  [0.8514] Recognition of Regular Shapes in Satelite Images (cs.CV, 2010)
  [0.8445] Classification of Cell Images Using MPEG-7-influenced Descriptors (stat.AP, 2008)

Filtered: RL + cs.LG + year>=2021
  [0.7681] Domain Adaptation: Learning Bounds and Algorithms (cs.LG, 2023)
  [0.7651] On the Dual Formulation of Boosting Algorithms (cs.LG, 2023)

Filtered: RL + year<2015
  [0.8859] Time Hopping technique for faster reinforcement learning (cs.AI, 2011)
  [0.8718] Rollout Sampling Approximate Policy Iteration (cs.LG, 2008)

Local metric comparison:
  Cosine top-5 == Dot product top-5: True
  L2 top-5 == Cosine top-5: True
```

---

## Частина 4 -- Chunking

### Fixed-size vs Semantic

Реалізовано дві стратегії для 30 найдовших анотацій:
- **fixed-size**: 50 слів на чанк, overlap 10 слів
- **semantic**: речення групуються поки не набереться ~80 слів

Semantic чанки дають осмисленіші одиниці -- кожен чанк містить завершені думки. Fixed-size часто ріже речення посередині, і такий обрізок вже не несе повного сенсу. Ембеддинг обрізаного речення виходить "розмитим" -- модель не може нормально закодувати фрагмент без початку або кінця.

Overlap у fixed-size збільшує кількість чанків (кожне слово потрапляє в кілька чанків), але зате текст покритий повністю -- нема "мертвих зон" на межі чанків. Без overlap межі чанків створюють розриви, і якщо ключова фраза потрапила на стик двох чанків, жоден з них не матиме її повністю.

На тестових запитах semantic chunking стабільно давав вищі score у top-5 -- чанки точніше відповідають запиту бо містять цілісні ідеї.

### Вивід скрипту

```
Selected 30 papers, abstract lengths 1820-1932 chars
Fixed chunks: 234, Semantic chunks: 146

Vectors in 'arxiv-chunks-fixed': 234
Vectors in 'arxiv-chunks-semantic': 146

[FIXED] 'attention mechanism in neural networks'
  0.8189 | Un metodo adaptativo para el modelo Bidominio en electrocard | chunk#7
  0.7956 | A Deterministic Model for Analyzing the Dynamics of Ant Syst | chunk#5
[SEMANTIC] 'attention mechanism in neural networks'
  0.8118 | Un metodo adaptativo para el modelo Bidominio en electrocard | chunk#4
  0.8036 | Shannon-Kotel'nikov Mappings for Analog Point-to-Point Commu | chunk#3
```

---

## Частина 5 -- Гібридний пошук

### Порівняльна таблиця

| Запит | BM25 top-1 | Vector top-1 | Hybrid top-1 |
|---|---|---|---|
| BERT fine-tuning | (нема збігів) | Time Hopping technique for faster RL | Time Hopping technique for faster RL |
| Yann LeCun convolutional networks | Network error correction... convolutional | Using SLP Neural Network to Persian Handwritten | Convolutional codes from units in matrix |
| making computers understand human emotions | Modeling the Experience of Emotion | Identification of parameters underlying emotions | Modeling the Experience of Emotion |

### Аналіз

**Який метод виграв і чому.** На запиті "BERT fine-tuning" BM25 не знайшов нічого -- бо в датасеті переважно статті 2007-2010, а BERT з'явився в 2018. Vector search хоча б знайшов тематично схожі ML-статті. На "Yann LeCun convolutional networks" BM25 знаходить статті зі словом "convolutional" але вони про error-correcting codes, а не про нейромережі -- класична проблема keyword search. Vector search знайшов нейромережеві статті бо specter2 розуміє контекст. Зате на "emotions from text" обидва методи знайшли релевантні статті, і гібрид поєднав їх рейтинги через RRF.

**Унікальні документи в гібридному top-5.** RRF може піднімати документи які були на 6-7 позиції в обох методах окремо, але через подвійний "голос" вони набирають достатньо RRF-скору щоб потрапити в top-5 гібриду. Це документи яких нема в top-5 жодного з методів окремо.

**Параметр k в RRF.** Формула: score(d) = sum(1/(k + rank_i)). При малому k (наприклад 1) перші позиції домінують -- різниця між rank 1 і rank 5 величезна. При великому k (60, як в цій реалізації) ранжування "м'якше" -- різниця між позиціями менша, і більше шансів у документів які стабільно високо в обох списках але не на першому місці.

### Вивід скрипту

```
BM25 index built over 8000 docs

Query: 'BERT fine-tuning'
  BM25: (пусто - нема BERT в датасеті 2007-2010)
  Vector: [0.8835] Time Hopping technique for faster RL
  Hybrid: [0.0164] Time Hopping technique for faster RL

Query: 'making computers understand human emotions from text'
  BM25: [17.1080] Modeling the Experience of Emotion
  Vector: [0.8818] Identification of parameters underlying emotions
  Hybrid: [0.0325] Modeling the Experience of Emotion
  In hybrid but not in individual top-5: {paper_5673}
```

---

## Частина 6 -- Аналіз і висновки

### 6.1 Семантичний пошук vs BM25

BM25 працює добре коли в документах є точні слова з запиту. На "BERT fine-tuning" BM25 взагалі нічого не знайшов бо в датасеті (переважно 2007-2010 роки) просто нема статей зі словом "BERT". Vector search хоча б знайшов тематично близькі ML-статті. Це важливий урок -- BM25 повністю безпорадний коли термін відсутній в корпусі.

На запиті "Yann LeCun convolutional networks" BM25 знайшов статті де є слово "convolutional" але вони про error correction codes, а не про нейромережі. Vector search знайшов реально нейромережеві статті бо specter2 розуміє контекст слова, а не просто шукає збіг токенів.

Зате на "emotions" обидва методи знайшли релевантні статті, і гібрид дав найкращий результат -- поєднав рейтинги і витягнув документ якого не було в top-5 жодного з методів окремо.

### Розмір чанка

Малі чанки (10-15 слів) -- це по суті одне-два речення. Ембеддинг такого чанка дуже специфічний, і якщо запит точно про цю деталь -- чудово. Але для широких запитів маленькі чанки не працюють, бо контексту замало щоб зрозуміти про що стаття.

Великі чанки (500+ слів) -- це по суті весь abstract цілком. Тоді ембеддинг "розмивається" -- він кодує все одразу, і специфічні деталі тонуть в загальному сенсі. Пошук за конкретною деталлю дає слабший recall.

Для цього датасету 50-80 слів на чанк -- нормальний компроміс. Достатньо контексту щоб ембеддинг мав сенс, але не так багато щоб специфіка губилась.

### Що буде з невірною метрикою

Якщо створити індекс з metric="euclidean", а модель дає нормалізовані вектори -- порядок видачі не постраждає. Формула з частини 3 (||a-b||^2 = 2 - 2*cos(a,b) для одиничних векторів) показує чому. L2 стає монотонною функцією cosine, ранжування еквівалентне.

Але є практичний нюанс. Pinecone оптимізує внутрішні структури під конкретну метрику -- approximate nearest neighbor алгоритми різні для cosine і L2. Якщо поставив не ту метрику, можеш втратити трохи в швидкості або точності ANN на великих обсягах. На 8k це непомітно, але на мільйонах вже може грати роль. Плюс score який повертає Pinecone інтерпретується по-різному -- при cosine це 0..1 (більше = краще), при L2 це відстань (менше = краще), і легко наплутати якщо метрика не збігається з тим що очікуєш.

### 6.4 Обмеження Pinecone Starter

Безкоштовний тір:
- обмежена кількість індексів та до 100K векторів -- для 8000 статей вистачає, але для датасету 10M вже ні
- обмежена пропускна здатність API
- нема replicas

Для 10M статей: треба або Pinecone платний тір, або self-hosted рішення (Qdrant/Milvus). З Qdrant можна розгорнути кластер на кількох нодах, шардувати по категоріях або роках. Ще варіант -- зменшити розмірність ембеддингів (PCA або модель з меншим dim) щоб влізти в RAM, і тримати декілька Pinecone індексів на платному плані.

Для 10M наукових статей Qdrant в Docker виглядає як найбільш практичний вибір -- тримає великі обсяги, підтримує snapshot бекапи, і не потребує оплати за кожен запит.
