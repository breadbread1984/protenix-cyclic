#!/bin/bash

cd frontend
python3 -m pip install -r requirements.txt --break-system-packages
python3 main.py
