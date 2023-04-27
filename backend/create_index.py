from config import DATA_PATH, EMBEDDING_MODEL, INDEX_PATH
from icews_utils import ICEWSDataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def main():
    # TODO: Index more data
    icews_dataset = ICEWSDataset(dir_path=DATA_PATH, dataset_name='ICEWS14', filename='test', idx=[0, 7370])
    train, test = [], []
    for d in icews_dataset.data:
        if d[3] == '2014-12-31':
            test.append(d)
        else:
            train.append(d)
    
    data = [', '.join(d) for d in train]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    faiss = FAISS.from_texts(data, embeddings)
    faiss.save_local(INDEX_PATH)

    with open('test_quads.txt', 'w', encoding='utf-8') as f:
        for d in test:
            f.write(', '.join(d) + '\n')

if __name__ == "__main__":
    main()
