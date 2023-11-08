import torch
from nltk.corpus import stopwords
from transformers import BertTokenizer
import os
import string
from nltk.tag import pos_tag
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt


class TopClusUtils(object):

    def __init__(self):
        pretrained_lm = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {k:v for v, k in vocab.items()}

    def encode(self, docs, max_len=512):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    def create_dataset(self, dataset_dir, text_file, loader_name, max_len=512):
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = []
            for doc in corpus.readlines():
                content = doc.strip()
                docs.append(content)
            print(f"Converting texts into tensors.")
            input_ids, attention_masks = self.encode(docs, max_len)
            print(f"Saving encoded texts into {loader_file}")
            stop_words = set(stopwords.words('english'))
            filter_idx = []
            valid_pos = ["NOUN", "VERB", "ADJ"]
            for i in self.inv_vocab:
                token = self.inv_vocab[i]
                if token in stop_words or token.startswith('##') \
                or token in string.punctuation or token.startswith('[') \
                or pos_tag([token], tagset='universal')[0][-1] not in valid_pos:
                    filter_idx.append(i)
            valid_pos = attention_masks.clone()
            for i in filter_idx:
                valid_pos[input_ids == i] = 0
            data = {"input_ids": input_ids, "attention_masks": attention_masks, "valid_pos": valid_pos}
            torch.save(data, loader_file)
        return data

    def cluster_eval(self, label_path, emb_path, seed=42):
        labels = open(label_path).readlines()
        labels = np.array([int(label.strip()) for label in labels])
        n_clusters = len(set(labels))
        embs = torch.load(emb_path)
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        y_pred = kmeans.fit_predict(embs.numpy())
        nmi = normalized_mutual_info_score(y_pred, labels)
        print(f"NMI score: {nmi:.4f}")
        return

# topic_emb: (1, n) n = num dimentions
# word_embs: (k, n) k = topk
def dimension_reduction(topic_emb, word_embs):
    topic_emb_np = np.array(topic_emb)
    topic_emb_np = np.array([topic_emb_np])
    word_embs_np = np.array(word_embs)

    n = topic_emb_np.size
    orthogonal_complement = []

    for _ in range(n - 1):
        U, S, Vt = np.linalg.svd(topic_emb_np)
        result_row = Vt[-1, :]
        topic_emb_np = np.vstack([topic_emb_np, result_row])
        orthogonal_complement.append(result_row)
    orthogonal_complement = np.array(orthogonal_complement)
    dim_reduced_embs = np.matmul(orthogonal_complement, word_embs_np.transpose()).transpose()
    # normalize
    row_norms = np.linalg.norm(dim_reduced_embs, axis=1, keepdims=True)
    normalized_embs = dim_reduced_embs / row_norms
    return torch.tensor(normalized_embs)
    # print(orthogonal_complement)
    

if __name__ == '__main__':
    topic_emb = torch.tensor([0,0,1])
    word_embs = torch.tensor([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
    print(dimension_reduction(topic_emb, word_embs))


    
