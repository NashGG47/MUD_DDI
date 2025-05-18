# cut_columns.py
with open("train.cod", "r", encoding="latin-1") as infile, open("train.cod.cl", "w", encoding="utf-8") as outfile:
    for line in infile:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            outfile.write("\t".join(parts[3:]) + "\n")

