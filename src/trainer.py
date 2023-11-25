from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from nltk.corpus import stopwords
import string
from transformers import BertTokenizer
from model import TopClusModel
import os
from tqdm import tqdm
import argparse
from sklearn.cluster import KMeans
from utils import TopClusUtils
import numpy as np
from utils import dimension_reduction


class TopClusTrainer(object):

    def __init__(self, args):
        self.args = args
        pretrained_lm = 'bert-base-uncased'
        self.n_clusters = args.n_clusters
        self.model = TopClusModel.from_pretrained(pretrained_lm,
                                                  output_attentions=False,
                                                  output_hidden_states=False,
                                                  input_dim=args.input_dim,
                                                  hidden_dims=eval(args.hidden_dims),
                                                  n_clusters=args.n_clusters,
                                                  kappa=args.kappa)
        self.utils = TopClusUtils()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.latent_dim = eval(args.hidden_dims)[-1]
        tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
        self.vocab = tokenizer.get_vocab()
        self.inv_vocab = {k:v for v, k in self.vocab.items()}
        self.filter_vocab()
        self.data_dir = os.path.join("datasets", args.dataset)
        self.utils.create_dataset(self.data_dir, "texts.txt", "text.pt")
        data = self.load_dataset(self.data_dir, "text.pt")
        input_ids = data["input_ids"]
        attention_masks = data["attention_masks"]
        valid_pos = data["valid_pos"]
        self.data = TensorDataset(input_ids, attention_masks, valid_pos)
        self.batch_size = args.batch_size
        self.res_dir = f"results_{args.dataset}"
        os.makedirs(self.res_dir, exist_ok=True)
        self.log_files = {}

    # invalid words to be filtered out from results
    def filter_vocab(self):
        stop_words = set(stopwords.words('english'))
        self.filter_idx = []
        for i in self.inv_vocab:
            token = self.inv_vocab[i]
            if token in stop_words or token.startswith('##') or len(token) <=2 \
               or token in string.punctuation or token.startswith('['):
                self.filter_idx.append(i)

    def load_dataset(self, dataset_dir, loader_name):
        loader_file = os.path.join(dataset_dir, loader_name)
        assert os.path.exists(loader_file)
        print(f"Loading encoded texts from {loader_file}")
        data = torch.load(loader_file, map_location=self.device)
        return data

    # pretrain autoencoder with reconstruction loss
    def pretrain(self, pretrain_epoch=20):
        pretrained_path = os.path.join(self.data_dir, "pretrained.pt")
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained model from {pretrained_path}")
            trainer.model.ae.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        else:
            print(f"Pretraining autoencoder")
            sampler = RandomSampler(self.data)
            dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
            model = self.model.to(self.device)
            model.eval()
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr)
            for epoch in range(pretrain_epoch):
                total_loss = 0
                for batch_idx, batch in enumerate(tqdm(dataset_loader, desc=f"Epoch {epoch+1}/{pretrain_epoch}")):
                    optimizer.zero_grad()
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    max_len = attention_mask.sum(-1).max().item()
                    input_ids, attention_mask = tuple(t[:, :max_len] for t in (input_ids, attention_mask))
                    input_embs, output_embs = model(input_ids, attention_mask, pretrain=True)
                    loss = F.mse_loss(output_embs, input_embs)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                print(f"epoch {epoch}: loss = {total_loss / (batch_idx+1):.4f}")
            torch.save(model.ae.state_dict(), pretrained_path)
            print(f"Pretrained model saved to {pretrained_path}")

    # initialize topic embeddings via K-Means clustering in the spherical latent space
    def cluster_init(self):
        latent_emb_path = os.path.join(self.data_dir, "init_latent_emb.pt")
        model = self.model.to(self.device)
        if os.path.exists(latent_emb_path) and os.path.exists(latent_emb_path):
            print(f"Loading initial latent embeddings from {latent_emb_path}")
            latent_embs, freq = torch.load(latent_emb_path, map_location=self.device)
        else:
            sampler = SequentialSampler(self.data)
            dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
            model.eval()
            latent_embs = torch.zeros((len(self.vocab), self.latent_dim)).to(self.device)
            freq = torch.zeros(len(self.vocab), dtype=int).to(self.device)
            with torch.no_grad():
                for batch in tqdm(dataset_loader, desc="Obtaining initial latent embeddings"):
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    valid_pos = batch[2].to(self.device)
                    max_len = attention_mask.sum(-1).max().item()
                    input_ids, attention_mask, valid_pos = tuple(t[:, :max_len] for t in (input_ids, attention_mask, valid_pos))
                    latent_emb = model.init_emb(input_ids, attention_mask, valid_pos)
                    valid_ids = input_ids[valid_pos != 0]
                    latent_embs.index_add_(0, valid_ids, latent_emb)
                    freq.index_add_(0, valid_ids, torch.ones_like(valid_ids))
            latent_embs = latent_embs[freq > 0].cpu()
            freq = freq[freq > 0].cpu()
            latent_embs = latent_embs / freq.unsqueeze(-1)
            print(f"Saving initial embeddings to {latent_emb_path}")
            torch.save((latent_embs, freq), latent_emb_path)

        print(f"Running K-Means for initialization")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.args.seed)
        kmeans.fit(latent_embs.numpy(), sample_weight=freq.numpy())
        model.topic_emb.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        # save topic_emb
        # topic_emb_path = os.path.join(self.res_dir, "topic_emb.pt")
        # print(f"Saving topic embeddings to {topic_emb_path}")
        # torch.save(model.topic_emb, topic_emb_path)


    # obtain topic discovery results and latent document embeddings for clustering
    def inference(self, topk=10, suffix=""):
        sampler = SequentialSampler(self.data)
        dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
        model = self.model.to(self.device)
        model.eval()
        latent_doc_embs = []
        # {word_id : latent_imbedding}
        latent_word_emb_dict = {}
        word_topic_sim_dict = defaultdict(list)
        b = 0
        with torch.no_grad():
            for batch in tqdm(dataset_loader, desc="Inference"):
                # shape: (32,512)
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                max_len = attention_mask.sum(-1).max().item()
                input_ids, attention_mask = tuple(t[:, :max_len] for t in (input_ids, attention_mask))
                # print(input_ids.shape)
                # print(input_ids)
                latent_doc_emb, latent_word_embs, word_ids, sim = model.inference(input_ids, attention_mask)
                latent_doc_embs.append(latent_doc_emb.detach().cpu())
                for latent_word_emb, word_id, s in zip(latent_word_embs, word_ids, sim):
                    word_topic_sim_dict[word_id.item()].append(s.cpu().unsqueeze(0))
                    if word_id.item() not in latent_word_emb_dict:
                        latent_word_emb_dict[word_id.item()] = latent_word_emb
                # if b == 1:
                #     break
                b += 1
        # len: 30522     
        # print(len(self.vocab))
        word_topic_sim = -1 * torch.ones((len(self.vocab), self.n_clusters))
        # iterate through all words in vocabulary, i is word id
        for i in range(len(word_topic_sim)):
            # if the word appears in all the documents more than 5 times
            if len(word_topic_sim_dict[i]) > 5:
                word_topic_sim[i] = torch.cat(word_topic_sim_dict[i], dim=0).mean(dim=0)
            else:
                if i in latent_word_emb_dict:
                    del latent_word_emb_dict[i]
        word_topic_sim[self.filter_idx, :] = -1

        # better organized topic display
        topic_sim_mat = torch.matmul(model.topic_emb, model.topic_emb.t())
        # print(topic_sim_mat)
        cur_idx = torch.randint(len(topic_sim_mat), (1,))
        topic_file = open(os.path.join(self.res_dir, f"topics{suffix}.txt"), "w")
        latent_word_emb_list = {}
        id_list = []
        for i in range(len(topic_sim_mat)):
            sort_idx = topic_sim_mat[cur_idx].argmax().cpu().numpy()
            # print(sort_idx, cur_idx)
            _, top_idx = torch.topk(word_topic_sim[:, sort_idx], topk)
            result_string = []
            for idx in top_idx:
                result_string.append(f"{self.inv_vocab[idx.item()]}")
            topic_list = [latent_word_emb_dict[idx.item()] for idx in top_idx]
            id_list.append(top_idx)
            topic_tensor = torch.stack(topic_list, dim=0)
            latent_word_emb_list[int(sort_idx)]=(top_idx, topic_tensor) 
            topic_file.write(f"Topic {int(sort_idx)}: {','.join(result_string)}\n")
            topic_sim_mat[:, sort_idx] = -1
            cur_idx = sort_idx

        latent_word_emb_list=sorted(latent_word_emb_list.items())
        # print(latent_word_emb_list[0])
        word_id_list = []
        word_emb_list = []
        for i in range(len(latent_word_emb_list)):
            word_id_list.append(latent_word_emb_list[i][1][0])
            word_emb_list.append(latent_word_emb_list[i][1][1])
        # latent_word_emb_list = [e for _,w,e,t in latent_word_emb_list]
        # word_id_list = [w for _,w,e in latent_word_emb_list]
        latent_doc_embs = torch.cat(latent_doc_embs, dim=0)
        doc_emb_path = os.path.join(self.res_dir, "latent_doc_emb.pt")
        print(f"Saving document embeddings to {doc_emb_path}")
        torch.save(latent_doc_embs, doc_emb_path)

        word_emb_path = os.path.join(self.res_dir, "latent_word_clusters_emb.pt")
        latent_word_clusters = torch.stack(word_emb_list, dim=0)
        # print(latent_word_clusters.shape)
        torch.save(latent_word_clusters, word_emb_path)
        print(f'Saving latent word embedding clusters to {word_emb_path}')


        word_id_path = os.path.join(self.res_dir, "latent_word_cluster_id.pt")
        latent_word_ids = torch.stack(word_id_list, dim=0)
        # print(latent_word_ids.shape)
        torch.save(latent_word_ids, word_id_path)
        print(f'Saving word id list to {word_id_path}')
        # shape (100, k, 100) - 100 topics, each topic has k word embeddings, each word embedding ias 100 dimension
        # print(latent_word_clusters.shape)

        
        return 

    # compute target distribution for distinctive topic clustering
    def target_distribution(self, preds):
        targets = preds**2 / preds.sum(dim=0)
        targets = (targets.t() / targets.sum(dim=1)).t()
        return targets

    # train model with three objectives
    def clustering(self, epochs=20):
        self.pretrain(pretrain_epoch=self.args.pretrain_epoch)
        self.cluster_init()
        sampler = RandomSampler(self.data)
        dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
        model = self.model.to(self.device)
        model.eval()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr)
        for epoch in range(epochs):
            total_rec_loss = 0
            total_rec_doc_loss = 0
            total_clus_loss = 0
            for batch_idx, batch in enumerate(tqdm(dataset_loader, desc=f"Clustering epoch {epoch+1}/{epochs}")):
                optimizer.zero_grad()
                # shape: (32, 512)
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                valid_pos = batch[2].to(self.device)
                max_len = attention_mask.sum(-1).max().item()
                input_ids, attention_mask, valid_pos = tuple(t[:, :max_len] for t in (input_ids, attention_mask, valid_pos))

                doc_emb, input_embs, output_embs, rec_doc_emb, p_word = model(input_ids, attention_mask, valid_pos)
                
                rec_loss = F.mse_loss(output_embs, input_embs)          # Lpre for f, g
                rec_doc_loss = F.mse_loss(rec_doc_emb, doc_emb)         # Lrec
                targets = self.target_distribution(p_word).detach()     # q(tk|zi(w))
                clus_loss = F.kl_div(p_word.log(), targets, reduction='batchmean')      # Lclus
                loss = rec_loss + rec_doc_loss + self.args.cluster_weight * clus_loss
                total_rec_loss += rec_loss.item()
                total_clus_loss += clus_loss.item()
                total_rec_doc_loss += rec_doc_loss.item()
                loss.backward()
                optimizer.step()
            # if (epoch+1) % 10 == 0 and self.args.do_inference:
            #     self.inference(topk=self.args.k, suffix=f"_{epoch}")
            print(f"epoch {epoch+1}: rec_loss = {total_rec_loss / (batch_idx+1):.4f}; rec_doc_loss = {total_rec_doc_loss / (batch_idx+1):.4f}; cluster_loss = {total_clus_loss / (batch_idx+1):.4f}")

        model_path = os.path.join(self.data_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"model saved to {model_path}")

    def sub_kMeans(self):
        cluster_ids = torch.load("./results_yelp/latent_word_cluster_id.pt", map_location=self.device)
        cluster_emb = torch.load("./results_yelp/latent_word_clusters_emb.pt", map_location=self.device)
        print("load data successfully")
        topic_emb = self.model.topic_emb
        # print(self.model.topic_emb.shape)
        # torch.Size([100, 100])
        # print(cluster_ids.shape)
        # torch.Size([100, 10])
        # print(cluster_emb.shape)
        # torch.Size([100, 10, 100])
        sub_topic_emb = []
        redu_cluster_emb = []
        ori_sub_topic_emb = []
        for i in tqdm(range(len(topic_emb)), desc="Reduction"):
            redu_cluster_emb.append(dimension_reduction(topic_emb[i].detach().cpu(),cluster_emb[i].cpu()))
            kmeans = KMeans(n_clusters=10, random_state=self.args.seed, n_init=10)
            kmeans2 = KMeans(n_clusters=10, random_state=self.args.seed, n_init=10)
            kmeans.fit(redu_cluster_emb[i].numpy())
            sub_topic_emb.append(torch.tensor(kmeans.cluster_centers_).to(self.device))
            kmeans2.fit(cluster_emb[i].detach().cpu().numpy())
            ori_sub_topic_emb.append(torch.tensor(kmeans2.cluster_centers_).to(self.device))
        sub_topic_emb = torch.stack(sub_topic_emb, dim=0).to(self.device)
        redu_cluster_emb = torch.stack(redu_cluster_emb, dim=0).to(self.device)
        ori_sub_topic_emb = torch.stack(ori_sub_topic_emb, dim=0).to(self.device)
        # print(sub_topic_emb.shape)
        # torch.Size([100, 3, 99])
        # print(redu_cluster_emb.shape)
        # torch.Size([100, 10, 99])

        topic_file = open(os.path.join(self.res_dir, f"sub_topics_Kmeans_10_comparison.txt"), "w")
        ori_topic_file = open(os.path.join(self.res_dir, f"ori_sub_topics_Kmeans_10_comparison.txt"), "w")
        # calculate the sim between the sub_topic_emb and the redu_cluster_emb
        for i in tqdm(range(len(topic_emb)), desc="Inference"):
            normalized_sub_topic_emb = F.normalize(sub_topic_emb[i], dim=-1)
            # torch.Size([3, 99])
            normalized_redu_cluster_emb = F.normalize(redu_cluster_emb[i], dim=-1)
            # torch.Size([10, 99])
            normalized_ori_sub_topic_emb = F.normalize(ori_sub_topic_emb[i], dim=-1)
            normalized_ori_cluster_emb = F.normalize(cluster_emb[i], dim=-1)
            sim = torch.matmul(normalized_redu_cluster_emb, normalized_sub_topic_emb.t())
            ori_sim = torch.matmul(normalized_ori_cluster_emb, normalized_ori_sub_topic_emb.t())
            # torch.Size([10,3])
            for j in range(len(normalized_sub_topic_emb)):
                _, top_idx = torch.topk(sim[:,j], 5)
                top_idx = torch.tensor(top_idx)
                id_list = cluster_ids[i][top_idx]
                result_string = []
                for idx in id_list:
                    result_string.append(f"{self.inv_vocab[idx.item()]}")
                topic_file.write(f"Topic {i}_{j}: {','.join(result_string)}\n")

            for j in range(len(normalized_ori_sub_topic_emb)):
                _, top_idx = torch.topk(ori_sim[:,j], 5)
                top_idx = torch.tensor(top_idx)
                id_list = cluster_ids[i][top_idx]
                result_string = []
                for idx in id_list:
                    result_string.append(f"{self.inv_vocab[idx.item()]}")
                ori_topic_file.write(f"Topic {i}_{j}: {','.join(result_string)}\n")
        return 




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='yelp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_clusters', default=100, type=int, help='number of topics')
    parser.add_argument('--k', default=10, type=int, help='number of top words to display per topic')
    parser.add_argument('--input_dim', default=768, type=int, help='embedding dimention of pretrained language model')
    parser.add_argument('--pretrain_epoch', default=20, type=int, help='number of epochs for pretraining autoencoder')
    parser.add_argument('--kappa', default=10, type=float, help='concentration parameter kappa')
    parser.add_argument('--hidden_dims', default='[500, 500, 1000, 100]', type=str)
    parser.add_argument('--do_cluster', action='store_true')
    parser.add_argument('--do_inference', action='store_true')
    parser.add_argument('--cluster_weight', default=0.1, type=float, help='weight of clustering loss')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs for clustering')

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = TopClusTrainer(args)

    trainer.sub_kMeans()
    
    # if args.do_cluster:
    #     trainer.clustering(epochs=args.epochs)
    # if args.do_inference:
    #     model_path = os.path.join("datasets", args.dataset, "model_100.pt")
    #     try:
    #         trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
    #     except:
    #         print("No model found! Run clustering first!")
    #         exit(-1)
    #     trainer.inference(topk=args.k, suffix=f"_final")
