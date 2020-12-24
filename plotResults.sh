# Plot all the data in the data directory
for file in data/*.csv; do
  echo "${file##*/}"
  matlab -batch 'plotResults("'${file##*/}'")'
done
