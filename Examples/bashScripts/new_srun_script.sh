#!/bin/env bash
helixType="C"
bendC=1.0
cS=6000.0
HpW=0.5
WpW=0.25

#  ---- EDIT *.py  ----
program="Coordinate_Minimizer_New.py"
outDir="Run/Helix_${helixType}_cS${cS}_H${HpW}_W${WpW}/"


mkdir -p ${outDir}


## run hostname for instance
##srun 
python3.10 ${program} ${helixType} ${bendC} ${cS} ${HpW} ${WpW} 2>&1 | tee ${outDir}out.txt 

