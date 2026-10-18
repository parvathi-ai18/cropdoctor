# CropDoctor

Upload a leaf photo and the app predicts the plant disease.

- Crops: tomato, potato, bell pepper (15 classes)
- Model: MobileNetV2 transfer learning, fine-tuned
- Dataset: PlantVillage
- Validation accuracy: 94.5% (lab-style photos)

## Limitation
Trained on PlantVillage lab photos (single leaf, plain background). Accuracy drops on real field photos. Future work: collect real field data.

## Run
Python 3.11 is required (TensorFlow does not support 3.14 yet).

    py -3.11 -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py
