import sys
import os
import argparse

from pathlib import Path

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

class TestVecs:
    def __init__(self, vecs, vec_words, vec_words_ix, 
                 test_words, test_words_ix, mul=False):
        sims = precompute_sims(vecs, vec_words_ix, test_words)
        if mul:
            # Enforce non-negativity on similarities.
            sims += 1
            sims /= 2
            self._score = self._mul
        else:
            self._score = self._add

        self.sims = sims
        self.vec_words = vec_words
        self.vec_words_ix = vec_words_ix
        self.test_words_ix = test_words_ix

    def _add(self, a, a_, b):
        return a_ - a + b

    def _mul(self, a, a_, b):
        num = a_ * b
        den = a + (a == 0) * 1e-10
        return num / den

    def run(self, a, a_, b):
        ix_a = self.test_words_ix[a]
        ix_a_ = self.test_words_ix[a_]
        ix_b = self.test_words_ix[b]

        scores = self._score(
            self.sims[:, ix_a],
            self.sims[:, ix_a_],
            self.sims[:, ix_b]
        )
        
        exclude_inputs = [self.vec_words_ix[w] for w in (a, a_, b)
                          if w in self.vec_words_ix]
        scores[exclude_inputs] = 0
        return self.vec_words[scores.argmax()]

    def test_set(self, test_pairs):
        results = []
        for a, a_ in test_pairs:
            for b, b_ in test_pairs:
                if a == b and a_ == b_:
                    continue
                b_ = set(b_)
                guess = self.run(a, a_[0], b)
                results.append(guess in b_)
        return results

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
    )
    parser.add_argument(
        'tests',
        help='The path to one or more test files',
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()

def test_groups(path_list):
    path_list = [Path(p) for p in sorted(path_list)]

    # If the only thing passed was a directory, and
    # it contains no test files (here assumed to be)
    # files ending with .txt, treat it as a list of 
    # its subdirectories.
    if len(path_list) == 1 and path_list[0].is_dir():
        contents = list(path_list[0].iterdir())
        if not any(p.suffix == '.txt' for p in contents):
            path_list = sorted(contents)

    groups = []
    for p in path_list:
        if p.is_dir():
            groups.append((p, sorted(p.glob('*.txt'))))
        elif p.is_file() and p.suffix == '.txt':
            groups.append((p, [p]))
    return groups

if __name__ == '__main__':
    args = get_args()

    vec_words, vec_words_ix, vecs = load_vecs(args.vectors)

    all_results = []
    for group, paths in test_groups(args.tests):
        group_results = []
        print()
        print('Test group {}'.format(group.stem))
        print()
        for path in paths:
            test_words, test_words_ix, test_pairs = load_test(path)
            test = TestVecs(vecs, vec_words, vec_words_ix, 
                            test_words, test_words_ix, mul=(args.method == 'mul'))

            results = test.test_set(test_pairs)

            group_results.extend(results)
            all_results.extend(results)
            print('{}: {} out of {} correct ({:.3g})'.format(
                Path(path).stem, sum(results), len(results), 
                sum(results) / len(results)
            ))

        # Only display stats for groups wtih two or more tests.
        if len(paths) > 1:
            print()
            print('Group {} results: {} out of {} correct ({:.4g})'.format(
                group.stem, sum(group_results), len(group_results), 
                sum(group_results) / len(group_results)
            ))
        print()
    print('All results: {} out of {} correct ({:.4g})'.format(
        sum(all_results), len(all_results), 
        sum(all_results) / len(all_results)
    ))
