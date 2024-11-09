# Modelo LSTM para Previsão de Séries Temporais

Esse repositório contém um código simples em Python para prever valores futuros com base em dados históricos usando uma Rede Neural LSTM (Long Short-Term Memory). 

O modelo foi desenvolvido usando a biblioteca Keras/TensorFlow e é projetado para prever variáveis temporais de séries históricas, como preços de ações ou qualquer dado que dependa do tempo.

## Funcionalidade

- **Carregamento de Dados**: O modelo carrega um arquivo CSV contendo dados históricos.
- **Pré-processamento**: Escala os dados usando o `MinMaxScaler` para garantir que os valores estejam dentro de um intervalo adequado.
- **Modelo LSTM**: A rede neural LSTM é treinada com os dados e prevê valores futuros.
- **Avaliação**: Exibe o gráfico comparando os dados reais com as previsões feitas pelo modelo.
