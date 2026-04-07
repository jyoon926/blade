import urllib.request
import gzip
import os
import shutil

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def download_mnist(root="data/mnist"):
    os.makedirs(root, exist_ok=True)
    for fname in FILES:
        out_path = os.path.join(root, fname[:-3])  # strip .gz
        if os.path.exists(out_path):
            print(f"Already exists: {out_path}")
            continue
        gz_path = os.path.join(root, fname)
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(MIRROR + fname, gz_path)
        print(f"Extracting ...")
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"  -> {out_path}")

if __name__ == "__main__":
    download_mnist()
    print("Done.")
