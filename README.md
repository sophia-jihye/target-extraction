# domain-specific target extraction

## `source`: Shortest path in dependency graph

## `source_archive/source_dp`: Double propagation rules 

#### Type 1 rules (given a set of seed opinion words -> new targets): R11, R12
> using opinion words to extract aspects (based on some dependency relations between them), **given a set of of seed opinion words a priori**. For example, specific rules that are instantiations of rule patterns **R11 and R12** are type 1 rules.

##### R11: `O(JJ) ~ MR <- T(NN)` or `O(JJ) -> MR ~ T`
- The phone has a good **screen**.
- The **photo quality** is amazing.
- The **software** of the player is not easy-to-use.
- I am not pleased with the **picture quality**. 
<img src="assets/R11.png"></img><img src="assets/R11a.png"></img><img src="assets/R11b.png"></img><img src="assets/R11c.png"></img>

##### R12: `O(JJ) ~ MR <- H -> MR ~ T(NN)`
- The **iPod** is the best mp3 player.
- The camera has a wonderful set of **features**. 
<img src="assets/R12.png"></img><img src="assets/R12a.png"></img>

#### Type 2 rules (using known targets -> new targets): R31, R32
> **using known aspects** to extract new aspects. The known aspects are extracted in the previous propagation. For example, specific rules that are instantiations of rule patterns **R31 and R32** are type 2 rules.

### Dataset annotation
* O: opinion words
    - POS tags for O: JJ, JJR, JJS
* T: targets
    - POS tags for T: NN, NNS
* H: any word
* O|T-Dep: Dependency relation
    - {MR}: mod, pnmod, subj, s, obj, obj2, desc
    - {CONJ}: conj

### Evaluation measures
* Precision and recall based on multiple occurences
    - <img src="assets/Mul_Precision,Recall.png"></img>

* Precision and recall based on distinct occurences
    - <img src="assets/Dis_Precision,Recall.png"></img>

* F1 score
    - <img src="assets/F1.png"></img>

### Reference
* Double propagation rules
    - [NinaTian98369](https://github.com/NinaTian98369/Double-propagation/blob/master/extract_targets_dp_new_final.py)
    - [opener-project (Java)](https://github.com/opener-project/double-propagation-target-generation/tree/master/src/main/java/org/openerproject/double_propagation2/algorithm/rules)
    - [opener-project (Java)](https://github.com/opener-project/double-propagation-target-generation/blob/master/src/main/java/org/openerproject/double_propagation2/model/RelationTypes.java)

* Customer review dataset
    - [uic.edu](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets)
* Customer review dataset parser (csv -> json)
    - [chakki-works](https://github.com/chakki-works/chazutsu/blob/7eea1f6b441db62ec76f64da1c041cb931746907/chazutsu/datasets/customer_review.py)
* Sentiment lexicon for customer reviews (Liu, 2004)
    - [uic.edu](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)

### Related papers
* Qiu, G., Liu, B., Bu, J., & Chen, C. (2011). Opinion word expansion and target extraction through double propagation. Computational linguistics, 37(1), 9-27.
* Liu, Q., Gao, Z., Liu, B., & Zhang, Y. (2015, June). Automated rule selection for aspect extraction in opinion mining. In Twenty-Fourth International Joint Conference on Artificial Intelligence.

<!--
## Obsolete
* MDSD dataset
    - https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
* MDSD datset parser (xml -> json)
    - https://github.com/robbymeals/word_vectors/blob/d829159e017695eb716413a02e3eee78fb86de25/src/mdsd2json.py
-->