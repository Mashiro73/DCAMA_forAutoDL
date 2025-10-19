@echo off
echo ========================================================
echo Starting Training for Fold 0
echo ========================================================

python ../train.py --datapath "../" ^
                --benchmark fssd12 ^
                --fold 0 ^
                --bsz 8 ^
                --nworker 4 ^
                --backbone swin ^
                --feature_extractor_path "..\backbones\swin_base_patch4_window12_384_22kto1k.pth"

echo ========================================================
echo Finished Training for Fold 0
echo ========================================================
pause

