#!/bin/bash
# Launch the Gradio UI for DocIntel Clinical Trial Knowledge Mining Platform

echo "🚀 Starting DocIntel Gradio UI..."
echo "   Access at: http://127.0.0.1:7860"
echo ""

cd "$(dirname "$0")"
pixi run python ui/app.py
