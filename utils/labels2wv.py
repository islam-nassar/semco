'''
Author: Islam Nassar
last edited: 24-Jul-2020
'''

import nltk
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pickle
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
import random


def get_labels2wv_dict(original_labels, wv_dict, return_mapping=False):
    """
    Receives a list of original labels and a word embedding model and returns a word_vector dictionary
    original_labels: list of original classes
    wv_dict: word vectors mapping dict or a string containing a pickle file path of such dict.
    return_mapping: if true, the mapping between original classes and new classes will be returned

    returns --> dictionary where keys are original labels and values are retrieved embeddings.
                optionally also returns mapping
    """
    if isinstance(wv_dict, str):
        print('Attempting to load Word Vectors from file..')
        wv_dict = pickle.load(Path(wv_dict).open('rb'))
    elif isinstance(wv_dict, dict):
        pass
    else:
        raise TypeError

    vocab_list = list(wv_dict.keys())
    vocab_set = set(vocab_list)
    labels2wv = {}
    lab_2_cur_lab = {}
    for label in original_labels:
        # standardise into word vectors standard (lower case with _ to separate words and ommit 'the_')
        curated_label = label.lower().replace('-', '_').replace(' ', '_').replace('the_', '').strip()

        if curated_label in vocab_set:
            pass
        else:
            # if not available in vocab, check the last bigram (ConceptNet have unigrams and bigrams only)
            words = curated_label.split('_')
            curated_label = '_'.join(
                words[-2:])  # even if it is a single token (ex: rollerskates), it won't throw an error

            if curated_label in vocab_set:
                print('last bigram in %s is found in vocab and will be used:' % label, curated_label)
                pass
            # to handle regular plural form, convert to singular form by removing s and check
            elif curated_label[:-1] in vocab_set:
                curated_label = curated_label[:-1]
                print('Singular form of %s is found in vocab and will be used: ' % label, curated_label)

            elif curated_label.replace('_', '') in vocab_set:
                curated_label = curated_label.replace('_', '')
                print('Collapsed %s is found in vocab and will be used: ' % label, curated_label)

            elif curated_label.replace('_', '')[:-1] in vocab_set:
                curated_label = curated_label.replace('_', '')[:-1]
                print('Collapsed singular form of %s is found in vocab and will be used: ' % label, curated_label)
            else:
                # # calculate edit distances between the source label and all possible words in vocab
                # # and select the word with the least edit distance as long as less than 3 edits were performed
                # print('Checking edit distances for label: ', label)
                # edit_distances = np.array([nltk.edit_distance(label, word) for word in vocab_list])
                # if 1 in edit_distances:
                #     curated_label = np.array(vocab_list)[np.argsort(edit_distances)[0]]
                #     print('Found similar word and label set to it: ', curated_label)

                # if no similar words, check if there is 'of' in the label and select the word before it to be your word (ex: hen of wooods --> hen)
                if 'of' in words:
                    print(label, ' contains of, replacing to: ', end="")
                    curated_label = words[words.index('of') - 1]
                    print(curated_label)
                else:
                    try:
                        pos_tag = nltk.pos_tag(words)
                    except LookupError:
                        nltk.download('averaged_perceptron_tagger')
                        pos_tag = nltk.pos_tag(words)
                    # if last word is a noun or a particle, choose it
                    if pos_tag[-1][1] in ['NN', 'NNS', 'RP']:
                        print('last word in %s is a noun, setting label to it: ' % label, end="")
                        curated_label = pos_tag[-1][0]
                        print(curated_label)
                    # otherwise choose the first noun in the composite label to be your label
                    else:
                        poses = [elem[1] for elem in pos_tag]
                        if 'NN' in poses:
                            curated_label = pos_tag[poses.index('NN')][0]

        if curated_label not in vocab_set:
            print('Checking edit distances for label: ', label)
            found = False
            for word in vocab_list:
                dist = nltk.edit_distance(label, word)
                if dist == 1:
                    found = True
                    curated_label = word
                    print('Found similar word and label set to it: ', curated_label)
                    break
            # edit_distances = np.array([nltk.edit_distance(label, word) for word in vocab_list])
            # if 1 in edit_distances:
            #     curated_label = np.array(vocab_list)[np.argsort(edit_distances)[0]]
            #     print('Found similar word and label set to it: ', curated_label)
            if not found:
                print('No similar words found, using last resort, random word choice:', end='')
                curated_label = random.choice(list(wv_dict.keys()))
                print(curated_label)

        # finally update the dictionary based on final curated label obtained
        lab_2_cur_lab[label] = curated_label
        labels2wv[label] = wv_dict[curated_label]

    if return_mapping:
        return labels2wv, lab_2_cur_lab
    return labels2wv


def show_embeddings(model, labels_list, plot_title='Before Grouping', group_assign=None):
    cmap = plt.cm.get_cmap('tab10')
    # fit a 2d PCA model to the vectors
    filtered_vocab = [k for k in model.keys() if k in labels_list]
    if group_assign is None:
        color_map =['black' for _ in range(len(filtered_vocab))]
    else:
        color_map = []
        for elem in filtered_vocab:
            for k,v in group_assign.items():
                if elem in v:
                    color_map.append(k)
                    break

        color_map = [cmap(i) for i in color_map]
    X = np.stack([model[word] for word in filtered_vocab])
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    plt.figure(figsize=(10, 10))
    plt.scatter(result[:, 0], result[:, 1], color=color_map)
    words = list(filtered_vocab)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), color=color_map[i], fontsize=13)

    plt.title(plot_title)
    plt.show()


def get_grouping(lab2wv_dict, eps=0.3, metric='cosine', return_mapping=False):
    """
    Gets label grouping after performing DBSCAN clustering.

    param lab2wv_dict: labels to word vectors dictionary (obtained using get_labels2wv_dict)

    param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    This is not a maximum bound on the distances of points within a cluster.
    (achieving reasonable clustering highly depends on dim of wvectors, 0.2 works well for 128D and 0.3 for 300D)

    param metric: metric used for similarity, options: 'cosine', 'l1', 'l2'
    param return_mapping: if set, the human readable form of grouping will also be returned

    :returns clust_assignment_labels (ndarray) and if 'return_mapping`, assignment dictionary

    """
    print('Grouping labels using DBscan....')
    keys = np.array(list(lab2wv_dict.keys()))
    vals = np.array(list(lab2wv_dict.values()))
    clust = DBSCAN(eps=eps, min_samples=1, metric=metric)
    clust.fit(vals)
    # fall out strategy to prevent the grouping from messing up. If grouping is aggressive, cancel it completely.
    if len(set(clust.labels_)) <= 5:
        print('Label grouping seems to have not work, aborting grouping.')
        clust.labels_ = list(range(len(clust.labels_)))

    print(f'{len(clust.labels_)} labels were grouped into {len(set(clust.labels_))} groups.')
    assign = {}
    for i, v in enumerate(clust.labels_):
        if v not in assign.keys():
            assign[v] = []
        assign[v].append(keys[i])

    for k, v in assign.items():
        if len(v) > 1:
            print(f'{v} are grouped together.')

    if not return_mapping:
        return clust.labels_
    return clust.labels_, assign


if __name__ == '__main__':
    from pathlib import Path
    import pickle

    classes = ['banded_gecko',
               'black_and_gold_garden_spider',
               'black_and_tan_coonhound',
               'wire_haired_fox_terrier',
               'soft_coated_wheaten_terrier',
               'west_highland_white_terrier',
               'german_short_haired_pointer',
               'greater_swiss_mountain_dog',
               "jack_o'_lantern",
               'polaroid_camera',
               'polaroid_land_camera',
               'hen_of_the_woods',
               'mushroom',
               'spider',
               'black-widow',
               'lord-of-the-rings',
               'cup-of-damascus',

               # 'tvmonitor',
               'rollerskates',
               'paint_can',
               'swing_set',
               'animal_migration',
               'great_wall_of_china',
               'power_outlet',
               'aquarium_fish']

    # classes = [str(i) for i in range (10)]

    # classes = pickle.load(Path('testing/cifar100_classes.pkl').open('rb'))

    classes = 'airplane automobile bird dog frog horse ship truck'.split() + \
              ['canine', 'doberman', 'wolf', 'car', 'vehicle', 'aircraft', 'jet',
               'boat']
    model = pickle.load(Path('numberbatch-en-19.08_128D.dict.pkl').open('rb'))
    labels2wv, label_mapping = get_labels2wv_dict(classes, model, return_mapping=True)
    clust_index, assign = get_grouping(labels2wv, eps=0.3, metric='cosine', return_mapping=True)

    show_embeddings(model, list(label_mapping.values()), plot_title='After grouping', group_assign=assign)
