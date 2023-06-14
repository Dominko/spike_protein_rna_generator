import gzip

from tqdm import tqdm

with gzip.open("datasets/raw/spikenuc0415_clean.fasta.gz", "wt") as out:
    with gzip.open("datasets/raw/spikenuc0415.fasta.gz", "rb") as handle:
        try:
            for line in tqdm(handle):
                line_1 = line.decode('utf-8','replace')
                out.write(line_1)
        except UnicodeDecodeError:
                print(line)
                print(line_1)
                pass
                # line=line.decode('utf-8','ignore').encode("utf-8")
                # out.write(line)

        handle.close()
        out.close()

# with gzip.open("data/test.txt.gz", "rb") as handle, open("data/test_2.txt", "wt") as out:
#     for line in tqdm(handle):
#         print(line)
#         line = line.decode('utf-8','replace')
#         print(line)
#         out.write(line)
#     handle.close()
#     out.close()

# with gzip.open("data/fuck_the_french.txt.gz", "rb") as handle, open("data/test_2.txt", "wt") as out:
#     for line in tqdm(handle):
#         print(line)
#         line = line.decode('ascii','replace')
#         print(line)
#         out.write(line)
#     handle.close()
#     out.close()