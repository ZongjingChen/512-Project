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
import torch.nn.functional as F
from tqdm import tqdm


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
        # print(topic_emb_np.shape)
        U, S, Vt = np.linalg.svd(topic_emb_np)
        result_row = Vt[-1, :]
        # print(Vt)
        topic_emb_np = np.vstack([topic_emb_np, result_row])
        orthogonal_complement.append(result_row)
    orthogonal_complement = np.array(orthogonal_complement)
    if np.any(orthogonal_complement == 0):
        print(orthogonal_complement)
    else:
        print('no 0')
    dim_reduced_embs = np.matmul(orthogonal_complement, word_embs_np.transpose()).transpose()
    if np.any(dim_reduced_embs == 0):
        print(dim_reduced_embs)
    else:
        print('no 0')
    # normalize
    row_norms = np.linalg.norm(dim_reduced_embs, axis=1, keepdims=True)
    if np.any(row_norms == 0):
        print(row_norms)
    else:
        print('no 0')
    normalized_embs = dim_reduced_embs / row_norms
    if np.any(normalized_embs == 0):
        print(normalized_embs)
    else:
        print('no 0')
    return torch.tensor(normalized_embs)
    # print(orthogonal_complement)
    

if __name__ == '__main__':
    
    # print(topic_emb)
    topic_emb = torch.tensor([0,0,1])
    # print(topic_emb.shape)
    word_embs = torch.tensor([[0,2,3],[2,3,4],[3,4,5],[4,5,6]])
    reduced_emb = dimension_reduction(topic_emb, word_embs)
    # latent_word_emb = torch.load('/home/zongjing/512-Project/results_yelp/latent_doc_emb.pt')
    # latent_word_emb = latent_word_emb[:, :]
    # print(latent_word_emb.shape)
    # n = 0
    # t = None
    # for _ in tqdm(range(1000), desc='test'):
    #     topic_emb = 2 * torch.rand((100)) - 1
    #     topic_emb = F.normalize(topic_emb, p=2, dim=0)
    #     reduced_emb = dimension_reduction(topic_emb, latent_word_emb)
    #     if torch.any(reduced_emb == 0).item():
    #         t = topic_emb
    #         break
    # print(t)
    # t = torch.tensor([ 0.0326, -0.0063, -0.1442,  0.0762, -0.1446,  0.0958, -0.1220,  0.0292,
    #      0.0605,  0.0762, -0.0385,  0.1031,  0.0084, -0.1627, -0.1427,  0.1207,
    #      0.0764,  0.0268, -0.1422,  0.0141,  0.0858,  0.0532, -0.0879, -0.1231,
    #      0.0696,  0.1178, -0.1320, -0.0570, -0.1028,  0.0837, -0.1646, -0.0323,
    #     -0.0889, -0.1021,  0.0302,  0.0631,  0.1472,  0.0180,  0.1338,  0.1651,
    #      0.0873,  0.0439, -0.0572, -0.0165, -0.1236, -0.1591, -0.1490, -0.0834,
    #      0.1513,  0.0682, -0.1501, -0.0382, -0.1611,  0.1465, -0.0183,  0.0843,
    #     -0.1579,  0.0412, -0.1004,  0.0924, -0.1275,  0.0262,  0.0720,  0.0363,
    #     -0.0662, -0.1058, -0.0181,  0.0865, -0.1331,  0.0568,  0.0146, -0.0488,
    #      0.0876, -0.1617,  0.1632, -0.1103, -0.1506,  0.0818,  0.1632, -0.0964,
    #      0.0781, -0.0996,  0.1116, -0.1081,  0.0397,  0.1053,  0.0798, -0.1586,
    #     -0.0308, -0.0337, -0.0026,  0.0120, -0.0376,  0.0264,  0.1357,  0.0851,
    #     -0.1585, -0.0812,  0.0676, -0.1204])
    # reduced_emb = dimension_reduction(t, latent_word_emb)
    # if torch.any(reduced_emb == 0).item():
    #     print('yes')
    # else:
    #     print('no')

    # print(n)



    
