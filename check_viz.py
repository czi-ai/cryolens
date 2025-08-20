#!/usr/bin/env python3
import sys

with open('/Users/kharrington/agent/git/czi-ai/cryolens/src/cryolens/visualization.py', 'r') as f:
    lines = f.readlines()[:20]
    
for i, line in enumerate(lines, 1):
    indent = len(line) - len(line.lstrip())
    print(f"Line {i:3d} [indent={indent:2d}]: {repr(line.rstrip())}")
