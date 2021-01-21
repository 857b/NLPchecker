PY=python3
CXX=g++
CPPFLAGS=-Wall -O2
CXXFLAGS=-std=gnu++11

.PHONY: preprocess datasets

preprocess: _data/vocab_dist.txt

datasets: _data/train/ _data/test/ _data/validation/

_data/vocab.txt:gen/export-vocab.py
	@mkdir -p _data
	@echo vocab
	@$(PY) gen/export-vocab.py > _data/vocab.txt

_build/edit_nn:gen/edit_nn.cpp
	@mkdir -p _build
	@echo CXX edit_nn
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $<

_data/vocab_dist.txt:_data/vocab.txt _build/edit_nn
	@echo vocab_dist
	@time ./_build/edit_nn gen/char_confusion.txt < _data/vocab.txt > $@

_data/train/:_data/vocab_dist.txt
	@echo generating training data
	@$(PY) cli.py --gen-data _data/train --src-data train --src-count 100000

_data/test/:_data/vocab_dist.txt
	@echo generating test data
	@$(PY) cli.py --gen-data _data/test --src-data test --src-count 1000

_data/validation/:_data/vocab_dist.txt
	@echo generating validation data
	@$(PY) cli.py --gen-data _data/validation --src-data validation \
		--src-count 1000
