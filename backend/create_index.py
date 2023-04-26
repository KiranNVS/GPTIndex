from config import DATA_PATH, EMBEDDING_MODEL, INDEX_PATH
from icews_utils import ICEWSDataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def main():
    # TODO: Index more data
    icews_dataset = ICEWSDataset(dir_path=DATA_PATH, dataset_name='ICEWS14', mode='test', idx=[0, 5000])
    data = [', '.join(d) for d in icews_dataset.data]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    faiss = FAISS.from_texts(data, embeddings)
    faiss.save_local(INDEX_PATH)

    test_data = ICEWSDataset(dir_path=DATA_PATH, dataset_name='ICEWS14', mode='test', idx=[5001, 5010])
    test_data = ['(' + ', '.join(d) + ')' for d in test_data.data]
    print(*test_data, sep='\n')

if __name__ == "__main__":
    main()
