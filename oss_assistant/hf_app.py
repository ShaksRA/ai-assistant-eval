"""
HuggingFace Spaces entry point — identical to oss_assistant/app.py
but with HF Spaces environment tweaks (no share=True needed).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Re-export the demo from app.py
from app import demo  # noqa

if __name__ == "__main__":
    demo.launch()
