# Literature Review

## Introduction
This document summarizes key papers on video-based autism screening using deep learning, with a focus on transfer learning and multi-modal approaches.

## Papers Summaries

# Related Work

Several studies have leveraged deep learning for video-based autism screening by combining spatial and temporal analysis. Early approaches have employed Convolutional Neural Networks (CNNs) alongside recurrent architectures such as Long Short-Term Memory (LSTM) networks. For instance, Washington *et al.* [@washington2021videobased] proposed a CNN+LSTM model that extracted spatial features from individual frames using a pre-trained CNN, followed by an LSTM to model temporal dynamics. This method demonstrated promising results in identifying behavioral markers of ASD from home videos.

Other research has explored the use of 3D CNNs to capture spatio-temporal features directly. Kojović *et al.* [@kojovic2021autism] utilized a 3D CNN architecture pre-trained on large-scale action recognition datasets to classify autism-related behaviors. Their work showed that transferring knowledge from extensive video datasets enables the model to capture subtle temporal patterns in children’s actions, thereby improving detection accuracy.

More recently, transformer-based models have emerged for video analysis. Park *et al.* [@park2024videotransformer] introduced a two-stream model that combines a CNN backbone with a Temporal Transformer encoder to capture long-range dependencies in video sequences. Similarly, Jaby *et al.* [@jaby2023asdevnet] developed an ensemble of Vision Transformers fine-tuned for ASD detection from facial expressions, achieving state-of-the-art performance by leveraging pre-trained models from large image corpora.

In addition, multimodal approaches have been investigated to fuse visual and audio cues along with explicit gaze tracking information. Natraj *et al.* [@natraj2024multimodal] combined audio features from pre-trained models with video features to enhance screening performance, while Chen *et al.* [@chen2024facialbehavior] demonstrated that integrating coarse and fine-grained facial behavior analyses using attention mechanisms leads to significant improvements.

## Deep Learning and Transfer Learning Approaches

Ding *et al.*'s systematic review and meta-analysis evaluated deep learning models for ASD prediction across 11 studies with 9,495 patients. Their analysis reported an overall sensitivity of 95%, specificity of 93%, and an AUC of 98%, underscoring the high diagnostic potential of these models. However, significant heterogeneity among studies—stemming from variations in study design, data quality, and model architecture—indicates that further standardized research is necessary before these models can be fully translated into clinical practice [@DingMeta].

Singh *et al.* proposed an innovative framework that screens ASD using home videos. Their method converts YouTube-sourced videos into skeletal keypoints to protect privacy and reduce computational complexity. A hybrid CNN-LSTM architecture, leveraging transfer learning with models such as MobileNet and Bi-LSTM, captures both spatial and temporal features. This approach achieved a test accuracy of approximately 84.95% with high precision, recall, and F1-scores [@SinghHomeVideos].

## Multimodal Systems and Behavioral Marker Analysis

Zhu *et al.* (2023) developed a multimodal machine learning system (MMLS) that quantifies toddlers' response-to-name—a key early indicator of autism—by analyzing video-recorded behavioral metrics such as response score, time, and duration. In a study involving 125 toddlers (with ASD, developmental delays, and typical development), the system achieved a computer-rated accuracy of 92% (AUC = 0.81), while human ratings yielded an AUC of 0.91 and diagnostic accuracy of 82.9%. These findings highlight the potential of automated video analysis for early screening, although reliance on a single behavioral marker may not be sufficient for a definitive diagnosis [@Zhu2023].

Dcouto and Pradeepkandhasamy reviewed multimodal deep learning approaches for early autism detection, emphasizing the integration of diverse data sources—such as neuroimaging, genetic information, and behavioral markers—to enhance diagnostic accuracy. Their work covers various architectures (including CNNs, DNNs, GCNs, and hybrid models) and discusses challenges related to data heterogeneity and model interpretability [@DcoutoPradeep].

## Computer Vision and Automated Behavioral Analysis

Hashemi *et al.* introduced a novel computer vision framework for quantifying autism risk behaviors. Their system analyzes dynamic facial expressions, gaze patterns, and head movements captured during structured behavioral tasks. By employing a self-contained mobile application that displays engaging movie stimuli to elicit responses such as name-calling and social referencing, the approach reliably captures subtle behavioral biomarkers. Validation against human-coded observations confirms its potential as a scalable and low-cost alternative to traditional screening methods [@HashemiCV].

Tariq *et al.* (2018) explored mobile-based ASD screening using classical machine learning techniques. In their study, non-expert raters manually annotated 30 behavioral features from home videos. Among eight classifiers, a sparse logistic regression model based on five key features (LR5) achieved a sensitivity of 94.5%, specificity of 77.4%, and overall accuracy near 89%—especially effective for children aged 2 to 6 years. However, manual feature annotation limits scalability [@Tariq2018].

Khan *et al.* (2022) compared several pre-trained CNN architectures (e.g., VGG, Inception, and Xception) for ASD detection using facial images. Their results indicated that the Xception model achieved the highest classification accuracy by effectively extracting deep facial morphological features. Nevertheless, the computational complexity and reliance on static features may restrict the model’s ability to capture dynamic behavioral cues critical for ASD assessment [@Khan2022].

The FIGS-DEAF model proposed by Saranya and Anandan integrates facial expressions with gait features using a fuzzy hybrid deep convolutional neural network. This model addresses overfitting and data imbalance issues and achieves up to 95% accuracy [@FIGSDEAF]. Additionally, Jarraya and Masmoudi introduced a hybrid 3D convolutional–transformer model for detecting stereotypical motor movements in autistic children during pre-meltdown crises. By combining a Temporal Vision Transformer (Temporal Swin) with a 3D ResNet backbone, their system captures long-range temporal dependencies and achieves 92% accuracy [@Jarraya2024].


