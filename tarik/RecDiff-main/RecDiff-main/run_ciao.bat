@echo off
echo Activating RecDiff environment and starting training on Ciao dataset...
echo Environment includes:
echo - Python=3.8
echo - torch=1.12.1
echo - numpy=1.23.1
echo - scipy=1.9.1
echo - dgl=1.0.2
echo.
cd /d "c:\Users\coren\Downloads\RecDiff-main\RecDiff-main"
"C:\Users\coren\miniconda3\envs\recdiff\python.exe" main.py --n_hid 64 --dataset ciao --n_layers 2 --s_layer 2 --lr 5e-3 --difflr 1e-3 --reg 1e-2 --batch_size 2048 --test_batch_size 1024 --emb_size 16 --steps 20 --noise_scale 1 --model_dir "./Models/ciao/"
pause
