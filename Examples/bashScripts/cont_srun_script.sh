#!/bin/env bash
helixType="C"
bendC=1.0
cS=6000.0
HpW=0.5
WpW=0.0

#  ---- EDIT *.py  ----
contValue=1000     #Continuation run number
program="Coordinate_Minimizer_Cont.py"


outDir="Run/Helix_${helixType}_cS${cS}_H${HpW}_W${WpW}_from${contValue}/"
mkdir -p ${outDir}



## run hostname for instance
##srun 
python3.7 ${program} ${helixType} ${bendC} ${cS} ${HpW} ${WpW} ${contValue} 2>&1 | tee ${outDir}out.txt 

