# Homework 04 — Decision Tree

Grew a classification tree with the `SimpleDecisionTreeClassifier` from the
course notes on 240 synthetic points in two dimensions, holding out
30% for testing.

## Results

| quantity | value |
| --- | --- |
| training accuracy | 0.9940 |
| test accuracy | 1.0000 |
| majority-class baseline | 0.5139 |
| leaves | 4 |
| depth | 3 |

## Reading the numbers

Training accuracy is 0.9940, not 1.0, because this tree has a stopping rule:
`MIN_SAMPLES_SPLIT = 40` turns any node with fewer than 40 rows into a leaf
labelled with its majority class, instead of splitting it further. One training
point (a class 0 point near x0 = 0.4, x1 = 2.1) sits on the wrong side of the
boundary, and the node holding it was too small to split again. The test
accuracy of 1.0000 is the number that says anything about generalisation.

The tree has 4 leaves at depth 3. Each leaf is a rectangle in
feature space, because every split compares one feature against one threshold.
That is why the decision regions have square corners, and why a tree needs a
staircase of splits to approximate a diagonal boundary that logistic regression
would capture with a single line.

## Effect of the stopping rule

| `MIN_SAMPLES_SPLIT` | leaves | depth | training accuracy | test accuracy |
| --- | --- | --- | --- | --- |
| 0 (grow to purity) | 5 | 4 | 1.0000 | 1.0000 |
| 5 | 5 | 4 | 1.0000 | 1.0000 |
| 40 | 4 | 3 | 0.9940 | 1.0000 |

Without the rule, the tree grows one extra leaf just to isolate that single
stray training point, which is memorising noise. With the rule at 40, the tree
gives up that one training point, loses a level of depth, and keeps the same
test accuracy. On these well-separated clusters the simpler tree costs nothing.
