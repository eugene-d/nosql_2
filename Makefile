PYTHON = .venv/bin/python

part_1: part_1_data part_1_embed

part_1_data:
	$(PYTHON) scripts/01_prepare_data.py

part_1_embed:
	$(PYTHON) scripts/02_embed.py

part_2:
	$(PYTHON) scripts/03_load_to_pinecone.py

part_3:
	$(PYTHON) scripts/04_search.py

part_4:
	$(PYTHON) scripts/05_chunking.py

part_5:
	$(PYTHON) scripts/06_hybrid_search.py

all: part_1 part_2 part_3 part_4 part_5
