#!/usr/bin/env python3
"""Use the baseline runner with explicit training-only semantic alignment."""
from opq_baseline import main

if __name__ == '__main__':
    main(require_alignment=True)
