"""Streamlit entry point for Cloud deployment.

Adds project root to path so src package imports work.
Run: streamlit run app.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.dashboard  # noqa: F401
