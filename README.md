# Fasti

![status](https://img.shields.io/badge/status-in%20development-orange)
![license](https://img.shields.io/badge/license-MIT-blue)

From Attention, Stance, and Topics to Ideal points.

Political actors reveal ideology through both the positions they take and the issues they emphasize, yet most text-based ideal-point measures capture only one of the two. Fasti is an item-response model that scales ideal points from issue salience and issue-specific stance jointly. It discovers topics inductively and classifies stance by transfer learning from the Manifesto Project, so it requires no human labels in the target corpus. A single latent factor generates both how much a document emphasizes an issue and which side it takes on it.

## Steps

1. **Sentence embeddings.** Split each document into sentences and encode each sentence into a contextual embedding vector.
2. **Topic modeling.** Reduce the dimensionality of the embeddings and cluster them by density into topics.
3. **Stance detection.** Classify each sentence as left, right, or neutral with a fine-tuned natural language inference transformer.
4. **Item-response model.** Estimate one ideal point per document from both the distribution of sentences across topics and the mean stance per topic.

The fitted parameters separate the issues that divide by stance from those that divide by salience.

## Repository

`code/` holds the pipeline: segmentation, embedding, topic modeling, stance training and inference, the joint fit, the comparison baselines, and the scripts that build the tables and figures. Scripts are numbered by run order. API keys are read from the environment and none are stored here.

## Citation

Leal, Jefferson L. "Which Issues Divide? Interpretable Ideal-Point Estimation from Political Text." Working paper.
