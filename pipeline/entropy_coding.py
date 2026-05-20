import pickle
import zlib
import os

def write_bitstream(data: dict, filepath: str):
    """
    Serialize, compress, and write bitstream data to a .bin file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    raw_bytes = pickle.dumps(data)
    compressed = zlib.compress(raw_bytes, level=9)
    with open(filepath, "wb") as f:
        f.write(compressed)

def read_bitstream(filepath: str) -> dict:
    """
    Read, decompress, and deserialize bitstream data from a .bin file.
    """
    with open(filepath, "rb") as f:
        compressed = f.read()
    data = pickle.loads(zlib.decompress(compressed))
    return data
