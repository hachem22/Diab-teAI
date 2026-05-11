@echo off
REM ============================================================
REM  Script de nettoyage final - DiabeteAI
REM  À lancer APRÈS avoir fermé PowerPoint
REM ============================================================

echo.
echo === Nettoyage final du projet DiabeteAI ===
echo.

REM Suppression des fichiers verrouillés et anciens
del /F /Q "Validation_Commerciale_MLA_Diabetes_Pro.pptx.pptx" 2>nul
del /F /Q "Validation_Commerciale_MLA_Diabetes_Pro_ENRICHED.pptx" 2>nul
del /F /Q "~$Validation_Commerciale_MLA_Diabetes_Pro.pptx.pptx" 2>nul
del /F /Q "~$Validation_Commerciale_MLA_Diabetes_Pro_ENRICHED.pptx" 2>nul
del /F /Q "~$Validation_Commerciale_MLA_Diabetes_Pro_FINAL.pptx" 2>nul

REM Renommage du PPTX final
if exist "Validation_Commerciale_MLA_Diabetes_Pro_FINAL.pptx" (
    ren "Validation_Commerciale_MLA_Diabetes_Pro_FINAL.pptx" "Validation_Commerciale_MLA_Diabetes_Pro.pptx"
    echo [OK] PPTX renomme : Validation_Commerciale_MLA_Diabetes_Pro.pptx
) else (
    echo [WARN] _FINAL.pptx introuvable - rien a renommer
)

REM Suppression de ce script lui-meme
del /F /Q "cleanup.bat" 2>nul

echo.
echo === Nettoyage termine ===
echo.
dir /B
echo.
pause
