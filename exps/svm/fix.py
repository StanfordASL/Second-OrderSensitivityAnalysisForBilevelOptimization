import pickle, gzip

fname = ["agd", "lbfgs", "sqp"]

result = dict()
for fname in fname:
    with gzip.open("data/logbarrier_opt_hist" + fname + ".pkl.gz", "rb") as fp:
        data = pickle.load(fp)
    key = list(data.keys())[0]
    result[key] = data[key]

with gzip.open("data/logbarrier_opt_hist.pkl.gz", "wb") as fp:
    pickle.dump(result, fp)
