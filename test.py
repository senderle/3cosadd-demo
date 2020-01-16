import numpy
import sys

def load_vecs(path):
    with open(path) as ip:
        rows = [line.split() for line in ip]
    words = [r[0] for r in rows]
    words_ix = {w: i for i, w in enumerate(words)}
    vecs = numpy.array([list(map(float, r[1:])) for r in rows])
    return words, words_ix, vecs

def load_test(path):
    with open(path) as ip:
        pairs = [line.split() for line in ip]
    words = [w for p in pairs for w in p]
    words_ix = {w: i for i, w in enumerate(words)}
    return words, words_ix, pairs

def precompute_sims(vecs, vec_words_ix, test_words):
    test_vecs = numpy.array([vecs[vec_words_ix[t]] for t in test_words])
    return vecs @ test_vecs.T

def perform_test(sims, vec_words, vec_words_ix, test_words_ix, a, a_, b):
    ix_a = test_words_ix[a]
    ix_a_ = test_words_ix[a_]
    ix_b = test_words_ix[b]

    scores = sims[:, ix_a_] - sims[:, ix_a] + sims[:, ix_b]
    scores[vec_words_ix[a]] = 0
    scores[vec_words_ix[a_]] = 0
    scores[vec_words_ix[b]] = 0

    return vec_words[scores.argmax()]

if __name__ == '__main__':
    vec_path = sys.argv[1] if len(sys.argv) > 1 else 'sample-vectors.txt'
    test_path = sys.argv[2] if len(sys.argv) > 2 else 'sample-test.txt'
    vec_words, vec_words_ix, vecs = load_vecs(vec_path)
    test_words, test_words_ix, test_pairs = load_test(test_path)

    sims = precompute_sims(vecs, vec_words_ix, test_words)

    results = []
    for a, a_ in test_pairs:
        for b, b_ in test_pairs:
            results.append(perform_test(sims, vec_words, vec_words_ix, test_words_ix, a, a_, b) == b_)
    print('{} out of {} correct'.format(sum(results), len(results)))
