# 3cosadd-demo
A quick demonstration of a fast 3CosAdd implementation

# Usage

With `numpy` installed, run

    python test.py

or, to use your own vector and test file, 

    python test.py [vectors.txt] [test1.txt] [test2.txt] ... [testN.txt]
    
**Note:** I wasn't sure how to handle tests with more than one answer, so results might not be correct in those cases. Right now, for a given fixed `a` and `b`, if *any* word in `a_` yields *any* word in `b_`, the test is marked as a pass. That seems excessively permissive, but I couldn't tell after a brief scan of `vecto` what the correct approach is.
