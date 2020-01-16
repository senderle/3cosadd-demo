import sys
import os
import argparse
import numpy

def load_vecs(path):
    with open(path) as ip:
        rows = [line.split() for line in ip]
    words = [r[0] for r in rows]
    words_ix = {w: i for i, w in enumerate(words)}
    vecs = numpy.array([list(map(float, r[1:])) for r in rows])
    return words, words_ix, vecs

def load_test(path):
    with open(path) as ip:
        pairs = [l.split() for l in ip]
        pairs = [[a, a_.split('/')] for a, a_ in pairs]
    words = [[a] + a_ for a, a_ in pairs]
    words = [w for aa_ in words for w in aa_]
    words_ix = {w: i for i, w in enumerate(words)}

    return words, words_ix, pairs

def precompute_sims(vecs, vec_words_ix, test_words):
    test_vecs = numpy.array([
        vecs[vec_words_ix[t]] if t in vec_words_ix else [0] * len(vecs[0]) 
             for t in test_words])
    return vecs @ test_vecs.T

def perform_test_add(sims, vec_words, vec_words_ix, test_words_ix, a, a_, b):
    ix_a = test_words_ix[a]
    ix_a_ = test_words_ix[a_]
    ix_b = test_words_ix[b]

    scores = sims[:, ix_a_] - sims[:, ix_a] + sims[:, ix_b]

    if a in vec_words_ix:
        scores[vec_words_ix[a]] = 0
    if a_ in vec_words_ix:
        scores[vec_words_ix[a_]] = 0
    if b in vec_words_ix:
        scores[vec_words_ix[b]] = 0

    return vec_words[scores.argmax()]

def perform_test_mul(sims, vec_words, vec_words_ix, test_words_ix, a, a_, b):
    ix_a = test_words_ix[a]
    ix_a_ = test_words_ix[a_]
    ix_b = test_words_ix[b]

    scores_num = sims[:, ix_a_] * sims[:, ix_b]
    scores_den = sims[:, ix_a] + (sims[:, ix_a] == 0) * 1e-10
    scores = scores_num / scores_den

    if a in vec_words_ix:
        scores[vec_words_ix[a]] = 0
    if a_ in vec_words_ix:
        scores[vec_words_ix[a_]] = 0
    if b in vec_words_ix:
        scores[vec_words_ix[b]] = 0

    return vec_words[scores.argmax()]

def get_args():
    parser = argparse.ArgumentParser(
        description='A simple 3CosAdd / 3CosMul analogy solver.'
    )

    parser.add_argument(
        '-m', 
        '--method', 
        default='add',
        choices=['add', 'mul'],
        type=str,
        help='The cosine offset solution method (additive vs. multiplicative).'
    )
    parser.add_argument(
        'vectors',
        help='The path to a text-based vector file',
        type=str,
        default='sample-vectors.txt'
    )
    parser.add_argument(
        'tests',
        help='The path to one or more test files',
        nargs=argparse.REMAINDER,
        default=['sample-test.txt']
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    perform_test = perform_test_add if args.method == 'add' else perform_test_mul

    vec_words, vec_words_ix, vecs = load_vecs(args.vectors)

    all_results = []
    for path in args.tests:
        test_words, test_words_ix, test_pairs = load_test(path)

        if any('/' in w for w in test_words):
            print('Tests with multiple answers are not supported yet. Skipping {}'.format(path))
            continue

        sims = precompute_sims(vecs, vec_words_ix, test_words)
        # Enforce non-negativity on similarities.
        if args.method == 'mul':
            sims = (sims + 1) / 2 

        results = []
        for a, a_ in test_pairs:
            for b, b_ in test_pairs:
                if a == b and a_ == b_:
                    continue

                b_ = set(b_)
                pair_results = []
                for w_a_ in a_:
                    guess = perform_test(sims, vec_words, vec_words_ix, test_words_ix, a, w_a_, b)
                    pair_results.append(guess in b_)
                results.append(any(pair_results))

        all_results.extend(results)
        filename = os.path.split(path)[-1]
        print('{}: {:.4g} out of {} correct ({:.4g})'.format(
            filename, sum(results), len(results), sum(results) / len(results)
        ))

    print('Collected results: {:.4g} out of {} correct ({:.4g})'.format(
        sum(all_results), len(all_results), sum(all_results) / len(all_results)
    ))
