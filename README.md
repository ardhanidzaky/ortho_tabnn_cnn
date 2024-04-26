# Development of Classification and Object Detection Models for Early Diagnosis in Orthodontics

## Author
*Ardhani Dzaky Ralfiano*
*Alfina Azaria*

Supervised by
*Ari Wibisono, S.Kom., M.Kom.*
*Prof. Dr. Ir. Petrus Mursanto, M.Sc.*

## Abstract
The COVID-19 pandemic has driven health transformation, especially in dental practice. The response to the risk of transmission leads the public towards telemedicine services,especially teledentistry. This phenomenon creates a new paradigm in orthodontics, encouraging the development of teleorthodontics. The support of machine learning technology in orthodontics offers innovative solutions for early diagnosis and increased accessibility to orthodontic services. This study will compare 3 computer vision models, which are EfficientNet, MobileNet, and ShuffleNet, accompanied by adding a tabular model, which is TabNet. The implementation of this computer vision model aims to provide an initial analysis for orthodontic patients and will be evaluated using the F1-score metric and expert interpretability with the help of LIME. This study found that the ShuffleNet computer vision model has the best average F1-score, followed by EfficientNet, and finally MobileNet. The difference in value ranges between 1-5% between EfficientNet and ShuffleNet, but the difference widens for MobileNet and ShuffleNet, which ranges between 3-8%. In addition, adding TabNet to the framework provides an average increase in F1-score by 2.7% to 5% compared to models that do not use TabNet.

## How to run
1. **Clone the Repository:** Clone this repository to your local machine.
2. **Install Dependencies:** Navigate to the project directory and install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

3. **Run the Code:** Execute the main script according to your requirements. Edit requirements on configs/config.yaml or create a new one.
```bash
python train.py --config configs/yaml
```

4. **Evaluation:** Evaluate the performance of the implemented algorithms or models from the logs that are saved under .logs/ folder.