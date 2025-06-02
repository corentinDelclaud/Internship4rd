@echo off
REM Script Windows pour lancer les entraînements et inférences

setlocal enabledelayedexpansion

REM Boucle sur les seeds (ici seulement 0, tu peux en ajouter d'autres si besoin)
for %%S in (0) do (

REM 1) inference only
REM a) Question-Only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0 --seed %%S
python inference.py --dataset webqsp --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0 --seed %%S

REM b) Textual Graph + Question
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name 7b_chat --seed %%S
python inference.py --dataset scene_graphs --model_name inference_llm --llm_model_name 7b_chat --seed %%S
python inference.py --dataset webqsp --model_name inference_llm --llm_model_name 7b_chat --seed %%S

REM 2) frozen llm + prompt tuning
REM a) prompt tuning
python train.py --dataset expla_graphs --model_name pt_llm --seed %%S
python train.py --dataset scene_graphs --model_name pt_llm --seed %%S
python train.py --dataset webqsp --model_name pt_llm --seed %%S

REM b) g-retriever
python train.py --dataset expla_graphs --model_name graph_llm --seed %%S
python train.py --dataset scene_graphs --model_name graph_llm --seed %%S
python train.py --dataset webqsp --model_name graph_llm --seed %%S

REM 3) tuned llm
REM a) finetuning with lora
python train.py --dataset expla_graphs --model_name llm --llm_frozen False --seed %%S
python train.py --dataset scene_graphs_baseline --model_name llm --llm_frozen False --seed %%S
python train.py --dataset webqsp_baseline --model_name llm --llm_frozen False --seed %%S

REM b) g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --seed %%S
python train.py --dataset scene_graphs --model_name graph_llm --llm_frozen False --seed %%S
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --seed %%S

)

pause