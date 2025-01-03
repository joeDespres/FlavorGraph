import pickle
from pandas import DataFrame


def main():
    file_path = "./output/kitchenette_embeddings.pkl"

    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)

    df = DataFrame(embeddings).T
    df.columns = [f"V{i+1}" for i in range(df.shape[1])]
    df.index.name = "flavor"
    df.reset_index(inplace=True)

    df.to_csv("./output/flavor_embeddings.csv", index=False)


if __name__ == "__main__":
    main()
