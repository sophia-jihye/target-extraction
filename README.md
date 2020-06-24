# domain-specific target extraction


## Double propagation rules
* O: opinion words
    - POS tags for O: JJ, JJR, JJS
* T: targets
    - POS tags for T: NN, NNS
* H: any word
* O|T-Dep: Dependency relation
    - {MR}: mod, pnmod, subj, s, obj, obj2, desc
    - {CONJ}: conj

##### Rule11: `O(JJ) ~ MR <- T(NN)` or `O(JJ) -> MR ~ T`
- The phone has a good **screen**.
- The **photo quality** is amazing.
- The **software** of the player is not easy-to-use.
- I am not pleased with the **picture quality**. 
<img src="assets/R11.png"></img><img src="assets/R11a.png"></img><img src="assets/R11b.png"></img><img src="assets/R11c.png"></img>

##### Rule12: `O(JJ) ~ MR <- H -> MR ~ T(NN)`
- The **iPod** is the best mp3 player.
- The camera has a wonderful set of **features**. 
<img src="assets/R12.png"></img><img src="assets/R12a.png"></img>

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