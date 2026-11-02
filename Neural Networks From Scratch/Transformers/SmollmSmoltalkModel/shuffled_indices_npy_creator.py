from pathlib import Path
import numpy as np

INPUT_NPY_PATH = "Data/tokenized/smoltalk_train_seq1025.npy"

# Set to None to automatically create:
# Data/tokenized/smollm_train_seq1025_order.npy
OUTPUT_ORDER_PATH = "Data/tokenized/smoltalk_train_order_e1.npy"

# Change this each epoch if you want a different shuffle.
# Use None for a non-deterministic random shuffle.
SEED = 86

def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    parent = input_path.parent
    return parent / f"{stem}_order.npy"

def create_shuffled_indices(input_path: Path, output_path: Path, seed: int | None):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    arr = np.load(input_path, mmap_mode="r")

    if arr.ndim < 1:
        raise ValueError(
            f"Expected input array with at least 1 dimension, got {arr.shape}"
        )

    num_sequences = arr.shape[0]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_sequences).astype(np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, indices)

    print("Done.")
    print(f"Input file:   {input_path}")
    print(f"Input shape:  {arr.shape}")
    print(f"Sequences:    {num_sequences:,}")
    print(f"Output file:  {output_path}")
    print(f"Output shape: {indices.shape}")
    print(f"Output dtype: {indices.dtype}")

    loaded = np.load(output_path, mmap_mode="r")

    print("Sanity check:")
    print(f"  min index:     {loaded.min()}")
    print(f"  max index:     {loaded.max()}")
    print(f"  expected max:  {num_sequences - 1}")
    print(f"  first 10:      {loaded[:10].tolist()}")


def main():
    input_path = Path(INPUT_NPY_PATH)

    if OUTPUT_ORDER_PATH is None:
        output_path = default_output_path(input_path)
    else:
        output_path = Path(OUTPUT_ORDER_PATH)

    create_shuffled_indices(
        input_path=input_path,
        output_path=output_path,
        seed=SEED,
    )


if __name__ == "__main__":
    main()