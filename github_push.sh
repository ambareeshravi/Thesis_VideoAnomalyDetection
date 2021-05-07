#!/bin/bash
ARG1=${1:-"Generic insignificant changes auto-push"}

git add *
git commit -m "$ARG1"
git push
cp AutoEncoders/VAD_AE_results.csv ../VAD_backup/