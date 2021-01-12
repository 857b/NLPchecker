PY=python3
CXX=g++
CPPFLAGS=-Wall -O2
CXXFLAGS=-std=gnu++11

.PHONY:preprocess

preprocess: _data/vocab_dist.txt

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
