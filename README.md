# Fasti: From Attention, Stance, and Topics to Ideal Points

![status](https://img.shields.io/badge/status-in%20development-orange)
![license](https://img.shields.io/badge/license-MIT-blue)

Political actors reveal ideology through both the positions they take and the issues they emphasize, yet most text-based ideal-point measures capture only one of the two. Fasti (**F**rom **A**ttention, **S**tance, and **T**opics to **I**deal Points) is an item response theory (IRT) model that jointly scales ideal points from issue salience and mean stance per issue. It discovers issues inductively and classifies stance with a natural language inference (NLI) transformer trained on data from the [Manifesto Project](https://manifesto-project.wzb.eu/), so it requires no human labels for scaling new texts.

## Steps

1. **Sentence embeddings.** Split each document into sentences and encode each sentence into a contextual embedding vector.
1. **Topic modeling.** Reduce the dimensionality of the embeddings and cluster them by density into topics, following BERTopic.
1. **Stance detection.** Classify each sentence as left, right, or neutral with a fine-tuned NLI transformer.
1. **IRT model.** Estimate one ideal point per document from both the distribution of sentences across topics and the mean stance per topic.

The fitted parameters can be interpreted as two measures:

* How much more salient each issue is to one ideological pole than to the other.
* How divisive each issue is in terms of left-right stance.

## Repository

`code/` holds the pipeline: segmentation, embedding, topic modeling, stance training and inference, the joint fit, the comparison baselines, and the scripts that build the tables and figures. Scripts are numbered by run order.

## Citation

Leal, Jefferson L. "Which Issues Divide? Interpretable Ideal-Point Estimation from Political Text." Working paper. [[Poster]](https://www.dropbox.com/scl/fi/q294x36auw4n0hy66ourr/Leal_Poster_Text_Ideology_Measure-Jefferson-Leal.pdf?rlkey=v4jvyrgmeboyup2p3cu5acc8x&dl=0)
