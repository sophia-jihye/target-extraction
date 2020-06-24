# domain-specific target extraction

## Double propagation rules
* O: opinion words
    - POS tags for O: JJ, JJR, JJS
* T: targets
    - POS tags for T: NN, NNS
* O|T-Dep: Dependency relation
    - {MR}: mod, pnmod, subj, s, obj, obj2, desc
    - {CONJ}: conj
<img src="assets/R11.png"></img>

## Reference
* Double propagation rules
    - https://github.com/NinaTian98369/Double-propagation/blob/master/extract_targets_dp_new_final.py
    - https://github.com/opener-project/double-propagation-target-generation/tree/master/src/main/java/org/openerproject/double_propagation2/algorithm/rules
    - https://github.com/opener-project/double-propagation-target-generation/blob/master/src/main/java/org/openerproject/double_propagation2/model/RelationTypes.java

* Customer review dataset
    - https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets
* Customer review dataset parser (csv -> json)
    - https://github.com/chakki-works/chazutsu/blob/7eea1f6b441db62ec76f64da1c041cb931746907/chazutsu/datasets/customer_review.py
* Sentiment lexicon for customer reviews (Liu, 2004)
    - https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon

## Related papers
* Qiu, G., Liu, B., Bu, J., & Chen, C. (2011). Opinion word expansion and target extraction through double propagation. Computational linguistics, 37(1), 9-27.
* Liu, Q., Gao, Z., Liu, B., & Zhang, Y. (2015, June). Automated rule selection for aspect extraction in opinion mining. In Twenty-Fourth International Joint Conference on Artificial Intelligence.

<!--
## Obsolete
* MDSD dataset
    - https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
* MDSD datset parser (xml -> json)
    - https://github.com/robbymeals/word_vectors/blob/d829159e017695eb716413a02e3eee78fb86de25/src/mdsd2json.py
-->