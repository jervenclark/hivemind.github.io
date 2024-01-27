## Overview
A decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has a hierarchical, tree structure which consists of a root node, branches, internal nodes and leaf nodes It is a way to display an algorithm that only contains conditional control statements.

![[static/images/decision-tree-e3d50161c5fc41bbae80778e0c2ad8f2.svg]]

As you can see from the diagram above, a decision tree starts with a root node, which does not have any incoming branches. The outgoing branches from the root node then feed into the internal nodes, also known as decision nodes. Based on the available features, both node types conduct evaluations to form homogeneous subsets, which are denoted by leaf nodes, or terminal nodes. The leaf nodes represent all possible outcomes within the dataset.

As an example, let's imagine you're trying to assess whether or not you should go surf, you may use the following decision rules to make a choice:

![[static/images/decision-tree-surfing-d2f5922803a347a9a0b66f924a6a337a.png]]

This type of flowchart structure also creates an easy to digest representation of decision making, allowing different groups across an organization to better understand why a decision was made.

Decision tree learning employs a divide and conquer strategy by conducting a greedy search to identify the optimal split points within a tree. This process of splitting is then repeated in a top-down, recursive manner until all, or the majority of records have been classified under specific class labels. Whether or not all data points are classified as homogeneous sets is largely dependent on the complexity of the decision tree. Smaller trees are more easily able to attain pure leaf nodes - i.e. data points in a single class. However, as a tree grows in size, it becomes increasingly difficult to maintain this purity, and it usually results in too little data falling within a given sub-tree. When this occurs, it is known as data fragmentation, and it can often lead to [[Overfitting|overfitting]]. As a result, decision trees have preference for small trees, which is consistent with the principle of parsimony in [[Occam's Razor]]; that is, "entities should add complexity only if necessary, as the simplest explanation is often the best."

The model's fit can then be evaluated through the process of cross validation. Another way that decision trees can maintain accuracy is by forming an ensemble via a [[Random Forest]] algorithm. This classifier predicts more accurate results, particularly when the individual trees are uncorrelated with each other.

## Types of Decision Trees
Decision trees can also be seen as [[Generative Models|generative models]] of induction rules from empirical data. An optimal decision tree is then defined as a tree that accounts for most of the data, while minimizing the number of levels (or questions). Several algorithms to generate such optimal trees have been devised such as:

- **ID3** - Ross Quinlan is credited within the development of [[Iterative Dichotomiser 3 (ID3)|ID3]], which is shorthand for [[Iterative Dichotomiser 3 (ID3)|Iterative Dichotomiser 3]]. This algorithm leverages entropy and information gain as metrics to evaluate candidate splits.
- **C4.5** - This algorithm is considered a later iteration if ID3, which was also developed by Quinlan. It can use information gain or gain ratios to evaluate split points within the decision trees.
- **CART** - The term [[Classification and Regression Trees (CART)|CART]], is an abbreviation for [[Classification and Regression Trees (CART)|classification and regression trees]] and was introduced by Leo Breiman. This algorithm typically utilizes [[Gini impurity]] to identify the ideal attribute to split on. Gini impurity measures how often a randomly chosen attribute is misclassified. When evaluating using Gini impurity, a lower value is more idea.

## Advantages and Disadvantages


## Notes
### Resources
- [[CHAID and Earlier Supervised Tree Methods]]
### Further Readings
- [[Induction of Decision Trees]]