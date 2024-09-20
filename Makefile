# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-autobuild
SOURCEDIR     = docs
BUILDDIR      = docs/html

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -h

docs:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: docs help Makefile
