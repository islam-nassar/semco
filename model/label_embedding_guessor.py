'''
author: Islam Nassar
date: 24-jul-2020
'''
import torch
import torch.nn.functional as F
import numpy as np

class LabelEmbeddingGuessor(object):

    def __init__(self, classes, label_group_idx, class_2_embeddings_dict, thresh, device):
        """

        :param classes: list of classes (this should be strictly in the same order of classes used throughout
        :param label_group_idx: the output of the label clustering operation (clust_labels_)
        :param class_2_embeddings_dict: output of labels2wv.get_labels2wv_dict
        :param thresh: an instance will be considered in the unlabelled loss if its label group threshold is greater
        than or equal thresh
        :param device: device

        """
        assert (all([cls in class_2_embeddings_dict for cls in classes])), 'label not found in emb dict'
        assert (len(classes) == len(label_group_idx)), 'classes length must match label_group_idx'

        self.classes = classes
        self.label_group_idx = torch.from_numpy(np.array(label_group_idx)).to(device)
        self.num_groups = len(torch.unique(self.label_group_idx))
        self.label_group_counts = self._get_label_groups_counts(label_group_idx).to(device)
        self.embedding_matrix = torch.tensor([class_2_embeddings_dict[cls] for cls in classes]).to(device)
        self.group_mask = self._get_group_mask().to(device)
        self.thresh = thresh
        self.device = device
        self.sharpening_factor = LabelEmbeddingGuessor.get_sharpening_factor(self.num_groups)

    def _get_group_mask(self):
        group_assign = np.zeros((self.num_groups, len(self.classes)))
        for i, item in enumerate(self.label_group_idx):
            group_assign[item, i] = 1
        group_mask = torch.tensor(group_assign, dtype=torch.float32)
        return group_mask

    def _get_label_groups_counts(self, label_idx):
        label_idx = list(label_idx)
        group_counts = [label_idx.count(cls) for cls in range(self.num_groups)]
        group_counts = torch.tensor(group_counts, dtype=torch.float32)
        return group_counts

    @staticmethod
    def get_sharpening_factor(num_groups):
        """
        Receives number of label groups and returns the recommended sharpening factor to achieve the desired performance
        (i.e. when classifier is confident about one class, softmax (applied on cos sim scores) returns 0.9 for such class).
        The formula in the function was obtained by applying linear regression on results of extensive monte carlo simulations
        of the problem. (refer to sharpening_factor.ipynb for more details)
        """
        # handle special case of low number of classes
        if num_groups < 10: return 6
        if 10 <= num_groups < 13: return 8 + (num_groups - 10) / 3
        elif 13 <= num_groups < 17: return 9 + (num_groups - 13) / 4

        # general case 
        coeffs = np.array([ 9.170736802898897, 8.61367218e-02, -2.14301427e-04,  2.64062187e-07, -1.48480191e-10,
            3.06355814e-14])
        feats = np.array([num_groups**i for i in range(6)])

        return coeffs @ feats.T

    def to(self, device):
        self.device = device
        self.label_group_idx = self.label_group_idx.to(device)
        self.label_group_counts = self.label_group_counts.to(device)
        self.embedding_matrix = self.embedding_matrix.to(device)
        self.group_mask = self.group_mask.to(device)




    def __call__(self, emb_preds):
        '''

        :param emb_preds: predictions obtained from the embedding head

        :return:
        '''
        assert (len(emb_preds.shape) > 1), 'LabelEmbedding Guessor only works on batches, unsqueeze if necessary'
        # obtain pairwise cosine similarity scores
        logits = F.cosine_similarity(emb_preds.unsqueeze(1), self.embedding_matrix.unsqueeze(0), dim=-1)
        # first obtain mask
        bs = emb_preds.size(0)
        del emb_preds
        # add up cosine sim scores for same label groups
        accum = torch.zeros(bs, self.num_groups, dtype=logits.dtype).to(self.device)
        accum.index_add_(-1, self.label_group_idx, logits)
        # divide by label group counts to normalise
        accum.div_(self.label_group_counts.unsqueeze(0))
        scores, idxs = torch.max(accum, -1)
        mask = scores.ge(self.thresh).float()
        del accum
        # then we calculate the embedding guess
        probs = F.softmax(logits, -1)
        # elementwise multiply with proper group mask to zero out other elements
        intermediate = probs * self.group_mask[idxs]
        # normalize again
        intermediate.div_(intermediate.sum(dim=1, keepdim=True))
        # obtain the weighted average of the embeddings as per initial prediction contribution
        embedding_guess = torch.matmul(intermediate, self.embedding_matrix)
        # finally capture idx_max logit (to help lower path if needed)
        _ , max_idx = torch.max(intermediate, -1)
        return embedding_guess, mask, scores, max_idx



if __name__ == '__main__':
    from utils.labels2wv import get_labels2wv_dict
    import random



    # # to debug insert a checkpoint in __call__ and override logits with below value
    small = (lambda: random.uniform(-1, 0.5))
    large = (lambda: random.uniform(0.6, 1))
    logits = torch.tensor([[large(), small(), small(), large(), small()],
                              [small(), large(), small(), small(), small()],
                              [small(), small(), small(), small(), small()]])


    labels = ['bicycle', 'apple', 'banana', 'motorbike', 'plane']
    clust_labels = [0, 1, 2, 0, 3]

    wv_dict_path = 'data/word_vectors/numberbatch-en-19.08_128D.dict.pkl'
    class_embs = get_labels2wv_dict(labels, wv_dict_path)
    emb_dim = len(list(class_embs.values())[0])

    lbemb_guessor = LabelEmbeddingGuessor(labels, clust_labels, class_embs, 0.6, 'cpu')

    emb_preds = torch.randn([3,emb_dim])
    emb_guess, mask, scores = lbemb_guessor(emb_preds)
    print('done')




