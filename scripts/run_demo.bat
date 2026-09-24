@echo off
cd /d "%~dp0.."
python scripts\generate_demo_data.py
python main.py --input data\input\demo_objects.png --method compare --ground-truth data\ground_truth\demo_objects_mask.png --seed 155,150
