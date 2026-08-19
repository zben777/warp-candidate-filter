




cd /home/bzhang/code/codex/warp-candidate-filter/scripts
./collect_profiles.sh --sudo-ncu


cd /home/bzhang/code/codex/warp-candidate-filter/scripts
./benchmark.sh 5



cd /home/bzhang/code/codex/warp-candidate-filter
git add -A && (git diff --cached --quiet || git commit -m "Update project") && git push origin main

