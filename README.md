---
layout: pubtex
title: Information-directed Learning
---

# Information-directed Learning

The corresponding thesis can be found [here](https://github.com/jonhue/masters-thesis).

## Getting started

### Installation

Requires Python 3.11.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### CI checks

* The code is auto-formatted using `black examples lib tests`.
* Static type checks can be run using `pyright`.
* Tests can be run using `pytest tests`.

### Documentation

To start a local server hosting the documentation run ```pdoc ./lib --math```.
