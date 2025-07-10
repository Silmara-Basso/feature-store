# feature-store
Gerenciamento de infraestrutura centralizada, com atualizações em tempo real para armazenamento e gerenciamento de features utilizadas em modelos de Machine Learning.


# Implementação de Feature Store em Pipeline de Deploy de Machine Learning


# Estrutura do Projeto:

feature-store/
├── dados/
│   ├── dados_brutos.csv
│   ├── feature_store.csv
│   ├── teste_features.csv
├── feature_store/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── feature_store.py
├── ml_pipeline/
│   ├── __init__.py
│   ├── model_training.py
│   ├── model_inference.py
│   ├── modelo_dsa.pkl
├── testes/
│   ├── test_feature_store.py
├── feature-store-main.py
├── feature-store-app.py
├── feature-store-cliente.py
├── requirements.txt


# Abra o terminal ou prompt de comando, navegue até a pasta com os arquivos e execute:
No meu caso tive que instalar o requests e setuptools também

````
pyenv local 3.11.5
````

````
python -m venv .pred
````

````
source .pred/bin/activate
````

''''
pip install -r requirements.txt
''''

''''
python -m unittest discover -s testes
''''

''''
python feature-store-main.py
''''

''''
python feature-store-app.py
''''


# Abra outro terminal ou prompt de comando, navegue até a pasta com os arquivos e execute:

''''
python feature-store-cliente.py
''''

````
deactivate
````