# Image-Caption-Generator

While humans can interpret these images without detailed captions, machines require some form of image captions for automatic understanding.

This project aims to develop an end-to-end solution for generating descriptive captions for images using deep learning techniques.

demo : https://the-caption-generator.streamlit.app/

## Dataset
- flickr dataset link :- https://www.kaggle.com/datasets/adityajn105/flickr8k
- I used the Flickr8k Dataset, which contains 8092 photographs and text descriptions. Dataset contain 5 caption for each Image

## Methodology for Image Captioning

### 1. Data Preprocessing
- Extract image features
- Text preprocessing
- Train-Test split
- Data generator

### 2. Encoder-Decoder Architecture
- Load VGG16 model
- Encoder : 
       Image feature layer
       Sequence feature layer
- Decoder


### 3. Training & Optimization
- Training model
- Evaluation of model

### 4. Frontend
- User interface using streamlit.
- ![image](https://github.com/user-attachments/assets/177f5a85-ccb5-4e75-a143-433743d0a349)



## Results
- The VGG16-LSTM model was trained for 20 epochs, achieving a low training loss of 2.1828.
- I evaluated the model using the BLEU score, with a focus on BLEU-1 score (0.536631).
- ![image](https://github.com/user-attachments/assets/b428ecdf-1e85-45f0-8474-855892088227)



#### Kaggle NoteBook :- https://www.kaggle.com/code/theshreyaa/image-captioning
