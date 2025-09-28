## TC-Sniffer: A Transformer-CNN Bibranch Framework Leveraging Auxiliary VOCs for Few-Shot UBC Diagnosis via Electronic Noses

> Authors: Yingying Jian, Nan Zhang, Yunzhe Bi, Xiyang Liu, Jinhai Fan*, Weiwei Wu*, Taoping Liu*
>
> Publication date: 2025/1/24
>
> Published in: ACS Sensors (vol. 10, no. 1, pp. 213-224, Jan. 2025)
>
> DOI: 10.1021/acssensors.4c02073
> 
> Abstract: Utilizing electronic noses (e-noses) with pattern recognition algorithms offers a promising noninvasive method for the early detection of urinary bladder cancer (UBC). However, limited clinical samples often hinder existing artificial intelligence (AI)-assisted diagnosis. This paper proposes TC-Sniffer, a novel bibranch framework for few-shot UBC diagnosis, leveraging easily obtainable UBC-related volatile organic components (VOCs) as auxiliary classification categories. These VOCs are biomarkers of UBC, helping the model learn more UBC-specific features, reducing overfitting in small sample scenarios, and reflecting the imbalanced distribution of clinical samples. TC-Sniffer employs intensity-based augmentation to address small sample size issues and focal loss to alleviate model bias due to the class imbalance caused by auxiliary VOCs. The architecture combines transformers and temporal convolutional neural networks to capture long- and short-range dependencies, achieving comprehensive representation learning. Additionally, feature-level constraints further enhance the learning of distinctive features for each class. Experimental results using e-nose data collected from a custom-designed sensor array show that TC-Sniffer significantly surpasses existing approaches, achieving a mean accuracy of 92.95% with only five UBC training samples. Moreover, the fine-grained classification results show that the model can distinguish between nonmuscle-invasive bladder cancer (NMIBC) and muscle-invasive bladder cancer (MIBC), both of which are subtypes of UBC. The superior performance of TC-Sniffer highlights its potential for timely and accurate cancer diagnosis in challenging clinical settings.

Description: The implementation of the ResNet and Transformer bi-branch encoder with efficient channel attention is made available.
