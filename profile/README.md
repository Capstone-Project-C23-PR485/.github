## Hi there 👋

# SkinCheck.AI | An All-in-One Platform For Your Skincare Needs. 


## SkinCheck.AI Android Applications


## SkinCheck.AI Cloud Computing Platform


## SkinCheck.AI Machine Learning Platform

Skin Condition Prediction and Skincare Recommendation

This repository contains the implementation of three machine learning models for predicting
different skin conditions, namely acne type, wrinkle detection, and flake detection. We
employed transfer learning and fine-tuning techniques using the MobileNetV2 architecture in
the TensorFlow framework. The models were trained on a custom dataset that was specifically
curated for these tasks.

### Skin Condition Prediction Models

#### Acne Type Prediction

The acne type prediction model utilizes deep learning techniques to classify images of skin
into different acne types. By training on a diverse set of images, the model can accurately
classify acne types, such as hormonal acne, comedonal acne, inflammatory acne, and cystic
acne. This model can assist dermatologists and skincare professionals in diagnosing and
treating various forms of acne.

#### Wrinkle Detection

The wrinkle detection model is designed to identify and localize wrinkles in facial images. By
leveraging the power of deep learning, this model can detect different types of wrinkles, such
as fine lines, crow's feet, and forehead wrinkles. It provides a useful tool for evaluating the
effectiveness of anti-aging treatments and skincare regimens.

#### Fleck Detection

The flake detection model aims to detect and analyze flaky skin conditions, such as dryness
and dandruff. By examining images of the skin surface, this model can identify areas with
flaking and provide insights into the severity of the condition. This information can aid skincare
professionals in recommending appropriate treatments and moisturizers for individuals with
flaky skin.

### Skincare Recommendation System

In addition to the skin condition prediction models, we have also developed a skincare
recommendation system. We created and preprocessed a comprehensive skincare product
dataset, which includes information about various skincare products. The dataset contains
details such as the brand, type of skincare, ingredients, suitability for different skin types, a
link to buy the product, and ratings.

Our skincare recommendation system leverages this dataset to generate personalized
skincare recommendations based on an individual's specific skin needs. By considering
factors such as skin type, concerns, and ingredient preferences, the system suggests suitable
skincare products that align with the user's requirements. This can simplify the process of
finding and selecting effective skincare products tailored to individual needs.

### Getting Started (for ML)

To get started with the skin condition prediction models and skincare recommendation system,
please follow the instructions below:
1. Clone this repository: 
    a) Acne:https://github.com/Capstone-Project-C23-PR485/Machine-Learning-Acne-TypeClassification.git
    b) Wrinkle:https://github.com/Capstone-Project-C23-PR485/-Machine-Learning-WrinkleType-Classification.git
    c) Flex:https://github.com/Capstone-Project-C23-PR485/Machine-Learning-Flex-TypeClassification.git
    d) Product Recommendation:https://github.com/Capstone-Project-C23-PR485/ProductRecommendation-.git
2. Install the required dependencies:
    a) pip install python
    b) pip install keras
    c) pip install tensorflow
3. Download the trained models and preprocessed datasets from the following links:
    a) Acne Type Prediction Model : https://drive.google.com/file/d/19y_LSalTBaCA8pk_Y0pa9RqldsEVj_Bz/view?usp=share_link
    b) Wrinkle Detection Model : https://drive.google.com/drive/folders/1Frk77G06bEWpdYkd7WNJ9tgl43ZkSeuS?us p=sharing
    c) Flake Detection Model : https://drive.google.com/file/d/18U8fijJX2el8s1GtWN_ngr58swDyYOzm/view?usp=sh aring
4. Place the downloaded models in the appropriate directories.
5. Explore the Jupyter notebooks provided in the repository to understand how to use the models and generate skincare recommendations.
6. Follow the instructions within the notebooks to load the models, preprocess the data, and make predictions or generate skincare recommendations.

### Fine-tuning the Model

To fine-tune the model, the MobileNetV2 architecture pretrained on ImageNet is used as the
base model. Fine-tuning starts from the 100th layer onwards. Layers before the 100th layer
are frozen, while layers after the 100th layer are trainable.

### Custom Layers
Custom layers are added on top of the base model to adapt it for the specific task. The added layers include:

1. Global Average Pooling
2. Batch Normalization
3. Dropout
4. Dense (with ReLU activation)
5. Dense (with sigmoid activation, kernel regularizer l2)

### Training and Evaluation

The model is compiled using the Adam optimizer with a learning rate of 0.0001 and categorical
cross-entropy loss. The evaluation metric used is accuracy. The training process involves
feeding the preprocessed images to the model and updating the model's parameters based
on the computed loss.

### Pretrained Model

The base model, MobileNetV2, is pretrained on the ImageNet dataset. This pretrained model
provides a good starting point for transfer learning, as it has already learned general features
from a large-scale dataset.

### Contributing

We welcome contributions to enhance the skin condition prediction models and skincare
recommendation system. If you have any ideas, bug fixes, or improvements, feel free to open
an issue or submit a pull.
