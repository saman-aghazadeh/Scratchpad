cat output_test.csv | awk -F '___' '{print $2}' > a.txt
cat output_test.csv | awk -F ',' '{print $2}' > b.txt
paste -d ',' a.txt b.txt > combined.txt
python calc_accuracy.py
