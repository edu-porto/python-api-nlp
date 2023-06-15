from flask import Flask, jsonify, request
import pickle
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt



# Abrindo o dicionário de dados que foram feitos o BoW
with open('dict_bow.pkl', 'rb') as dict_bow:
    dictionary_custom = pickle.load(dict_bow)

# Abrindo o modelo 
with open('modelo_naive_bayes.pkl', 'rb') as model_nb:
    model = pickle.load(model_nb)

# Dicionários de emoji 
emoji_dict = {
    '😀': 'feliz ',
    '😂': 'risos ',
    '😔': 'triste ',
    '👏': 'palmas ',
    '🥰': 'amável ',
    '💙': 'coração azul ',
    '🙏🏼': 'orando ',
    '✨': 'brilhando ',
    '🤮': 'nojo ',
    '🚀': 'foguete ',
    '👿': 'diabo ',
    '🤢': 'nojo ',
    '🔥': 'fogo ',
    '😡': 'fúria ',
    '😠': 'raiva ',
    '🤣': 'rindo ',
    '😃': 'feliz ',
    '😎': 'curtindo ',
    '😊': 'feliz ',
    '🤩': 'maravilhado ',
    '😋': 'delicioso ',
    '😆': 'risada ',
    '😌': 'calmo ',
    '🤔': 'pensativo ',
    '😷': 'máscara ',
    '🤣': 'muitoRiso ',
    '🥺': 'carinhoso ',
    '👍': 'positivo ',
    '🤯': 'menteExplodida ',
    '😅': 'alívio ',
    '🥰': 'carinhaComCoração ',
    '😓': 'suor ',
    '😑': 'tédio',
    '🤫': 'silêncio',
    '🤝': 'apertoDeMãos',
    '😊': 'sorriso',
    '😍': 'apaixonado',
    '😭': 'choro ',
    '🤗': 'abraço ',
    '🎉': 'festa ',
    '😎': 'descolado ',
    '😱': 'surpresa ',
    '😴': 'sono ',
    '🙌': 'celebração ',
    '🤔': 'pensativo ',
    '😘': 'beijo ',
    '🥳': 'festeiro ',
    '🙄': 'revirarOsOlhos ',
    '😌': 'alívio ',
    '🤫': 'segredo ',
    '😇': 'inocente ',
    '😂': 'muitoEngraçado ',
    '🤔': 'pensando ',
    '😴': 'sono ',
    '🤪': 'loucura ',
    '😢': 'decepcionadoAliviado ',
    '😬': 'nervoso ',
    '😌': 'alívio',
    '😔': 'triste ',
    '😞': 'desapontado ',
    '😢': 'choro ',
    '😭': 'chorando ',
    '😡': 'raiva ',
    '🤯': 'mente explodida ',
    '😳': 'surpreso ',
    '😱': 'gritando ',
    '😨': 'assustado ',
    '😴': 'sono ',
    '🥱': 'bocejando ',
    '🤢': 'enjoado ',
    '🤮': 'vomitando ',
    '🤧': 'espirro ',
    '🤒': 'doente ',
    '🤕': 'machucado ',
    '🤑': 'dinheiro ',
}

# Array de stop_words customizada 
stop_words_custom = [
            '@', 'banco', 'btg', 'brg', 'pactual', 'btgpactual', 'pq', 'q', 'pra', 'vcs', 'vc', 'i', 'p', 'kkk', 'y', 'of',
            'n', 'a', 'à', 'as', 'o', 'os', 'e', 'aos', 'do', 'das', 'dos', 'das', 'de', 'deles', 'dela', 'deles', 'delas',
            'para', 'que', 'em', 'algo', 'algum', 'alguma', 'alguns', 'algumas', 'aqui', 'aquele', 'aquela', 'aqueles',
            'aquelas', 'aqui', 'aquilo', 'cá', 'com', 'como', 'cada', 'coisa', 'daquele', 'daquela', 'daquilo', 'daqueles',
            'daquelas', 'desse', 'deste', 'dessa', 'desses', 'destes', 'destas', 'ele', 'eles', 'ela', 'elas', 'eu', 'nos',
            'nós', 'vocês', 'voces', 'enquanto', 'era', 'está', 'estamos', 'estão', 'estar', 'estará', 'estive', 'estivemos',
            'estiver', 'estivera', 'estiveram', 'estivéramos', 'estiverem', 'estivermos', 'estivesse', 'estivessem',
            'estivéssemos', 'estiveste', 'estivestes', 'estou', 'fará', 'farta', 'farto', 'fez', 'fim', 'foi', 'fomos',
            'for', 'fora', 'foram', 'fôramos', 'forem', 'formos', 'fosse', 'fossem', 'fôssemos', 'foste', 'fostes', 'fui',
            'fôssemos', 'há', 'houve', 'hoje', 'isso', 'isto', 'já', 'lá', 'lhe', 'lhes', 'lo', 'logo', 'mas', 'me', 'mesma',
            'mesmas', 'mesmo', 'mesmos', 'meu', 'meus', 'minha', 'minhas', 'na', 'no', 'nas', 'nos', 'naquela', 'naquelas',
            'naquele', 'naqueles', 'nem', 'nessa', 'nessas', 'nesse', 'nesses', 'nesta', 'nestas', 'neste', 'nestes',
            'ninguém', 'nosso', 'nossa', 'nossos', 'nossas', 'num', 'numa', 'outra', 'outras', 'outro', 'outros', 'pela',
            'pelo', 'perante', 'pois', 'ponto', 'pontos', 'por', 'porém', 'porque', 'porquê', 'própria', 'próprio',
            'próprias', 'próprios', 'qual', 'quando', 'quanto', 'quantos', 'quantas', 'quê', 'quem', 'quer', 'quereis',
            'querem', 'queremas', 'quis', 'quisemos', 'quiser', 'quisera', 'quiseram', 'quiséramos', 'quiserem',
            'quisermos', 'quisésseis', 'quiséssemos', 'quiseste', 'quisestes', 'quiseste', 'quisestes', 'quizer',
            'quizeram', 'quizerem', 'quizermos', 'quizesse', 'quizessem', 'quizéssemos', 'são', 'se', 'seja', 'sejam',
            'sejamos', 'sem', 'sendo', 'ser', 'será', 'serão', 'será', 'seriam', 'seríamos', 'serias', 'seríeis', 'sete',
            'seu', 'seus', 'sob', 'sobre', 'sois', 'só', 'somos', 'sou', 'sua', 'suas', 'tal', 'talvez', 'também', 'te',
            'tem', 'têm', 'temos', 'tendes', 'tenha', 'tenham', 'tenhamos', 'tenho', 'tens', 'ter', 'terá', 'terão',
            'terá', 'teriam', 'teríamos', 'terias', 'teríeis', 'teu', 'teus', 'teve', 'tivemos', 'tiver', 'tivera',
            'tiveram', 'tivéramos', 'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivéssemos', 'tiveste', 'tivestes',
            'tiveste', 'tivestes', 'um', 'uma', 'umas', 'uns' ]

# Definindo novas váriaveis para carregar o modelo e dict
dict_bow = dictionary_custom
model_bow = model

app = Flask(__name__)


# Funções para gerar a predição 

# Converte emojis em palavras 
def emoji_to_word(textos):
    textos_processados = []
    for texto in textos:
        palavras = texto.split()
        texto_processado = []
        for palavra in palavras:
            if palavra in emoji_dict:
                texto_processado.append(emoji_dict[palavra])
            else:
                texto_processado.append(palavra)
        texto_processado = ' '.join(texto_processado)
        textos_processados.append(texto_processado)
    
    return textos_processados

# Tokeniza o texto
def processarTexto(textos):
    textos_processados = []
    for texto in textos:
        texto = texto.lower()
        tokens = word_tokenize(texto)
        stop_words = [
            '@', 'banco', 'btg', 'brg', 'pactual', 'btgpactual', 'pq', 'q', 'pra', 'vcs', 'vc', 'i', 'p', 'kkk', 'y', 'of',
            'n', 'a', 'à', 'as', 'o', 'os', 'e', 'aos', 'do', 'das', 'dos', 'das', 'de', 'deles', 'dela', 'deles', 'delas',
            'para', 'que', 'em', 'algo', 'algum', 'alguma', 'alguns', 'algumas', 'aqui', 'aquele', 'aquela', 'aqueles',
            'aquelas', 'aqui', 'aquilo', 'cá', 'com', 'como', 'cada', 'coisa', 'daquele', 'daquela', 'daquilo', 'daqueles',
            'daquelas', 'desse', 'deste', 'dessa', 'desses', 'destes', 'destas', 'ele', 'eles', 'ela', 'elas', 'eu', 'nos',
            'nós', 'vocês', 'voces', 'enquanto', 'era', 'está', 'estamos', 'estão', 'estar', 'estará', 'estive', 'estivemos',
            'estiver', 'estivera', 'estiveram', 'estivéramos', 'estiverem', 'estivermos', 'estivesse', 'estivessem',
            'estivéssemos', 'estiveste', 'estivestes', 'estou', 'fará', 'farta', 'farto', 'fez', 'fim', 'foi', 'fomos',
            'for', 'fora', 'foram', 'fôramos', 'forem', 'formos', 'fosse', 'fossem', 'fôssemos', 'foste', 'fostes', 'fui',
            'fôssemos', 'há', 'houve', 'hoje', 'isso', 'isto', 'já', 'lá', 'lhe', 'lhes', 'lo', 'logo', 'mas', 'me', 'mesma',
            'mesmas', 'mesmo', 'mesmos', 'meu', 'meus', 'minha', 'minhas', 'na', 'no', 'nas', 'nos', 'naquela', 'naquelas',
            'naquele', 'naqueles', 'nem', 'nessa', 'nessas', 'nesse', 'nesses', 'nesta', 'nestas', 'neste', 'nestes',
            'ninguém', 'nosso', 'nossa', 'nossos', 'nossas', 'num', 'numa', 'outra', 'outras', 'outro', 'outros', 'pela',
            'pelo', 'perante', 'pois', 'ponto', 'pontos', 'por', 'porém', 'porque', 'porquê', 'própria', 'próprio',
            'próprias', 'próprios', 'qual', 'quando', 'quanto', 'quantos', 'quantas', 'quê', 'quem', 'quer', 'quereis',
            'querem', 'queremas', 'quis', 'quisemos', 'quiser', 'quisera', 'quiseram', 'quiséramos', 'quiserem',
            'quisermos', 'quisésseis', 'quiséssemos', 'quiseste', 'quisestes', 'quiseste', 'quisestes', 'quizer',
            'quizeram', 'quizerem', 'quizermos', 'quizesse', 'quizessem', 'quizéssemos', 'são', 'se', 'seja', 'sejam',
            'sejamos', 'sem', 'sendo', 'ser', 'será', 'serão', 'será', 'seriam', 'seríamos', 'serias', 'seríeis', 'sete',
            'seu', 'seus', 'sob', 'sobre', 'sois', 'só', 'somos', 'sou', 'sua', 'suas', 'tal', 'talvez', 'também', 'te',
            'tem', 'têm', 'temos', 'tendes', 'tenha', 'tenham', 'tenhamos', 'tenho', 'tens', 'ter', 'terá', 'terão',
            'terá', 'teriam', 'teríamos', 'terias', 'teríeis', 'teu', 'teus', 'teve', 'tivemos', 'tiver', 'tivera',
            'tiveram', 'tivéramos', 'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivéssemos', 'tiveste', 'tivestes',
            'tiveste', 'tivestes', 'um', 'uma', 'umas', 'uns'
        ]
        tokens = [
            token for token in tokens if token not in stop_words and not token.startswith('@') and token.isalpha()
        ]
        textos_processados.append(tokens)
    return textos_processados

#  Vetoriza as frases com o dicionário do BoW
def vetorizar_frases(frases, dictionary):
    frases = [' '.join(tokens) for tokens in frases]
    vectorizer = CountVectorizer()
    vectorizer.set_params(vocabulary=dictionary)
    frases_vetorizadas = vectorizer.fit_transform(frases)
    return frases_vetorizadas

# Rotas da api 

@app.route('/classificar', methods=['POST'])
def classificar():
    dados = request.json
    # Aplique a função emoji_to_word() aos dados do web scraping
    textos_entrada = dados["dados"]
    textos_processados = emoji_to_word(textos_entrada)
    textos_processados = processarTexto(textos_processados)
    frases_vetorizadas = vetorizar_frases(textos_processados, dict_bow)
    frases_vetorizadas = frases_vetorizadas.toarray()  # Converter para matriz densa
    predicoes = model_bow.predict(frases_vetorizadas)

    # Mapear valores numéricos para palavras correspondentes
    mapeamento_classes = {0: "negativo", 1: "neutro", 2: "positivo"}
    predicoes_palavras = [mapeamento_classes[predicao] for predicao in predicoes]

    return json.dumps(predicoes_palavras)

# Conta a proporção de quantos sentimentos há no json
@app.route('/proporcoes', methods=['POST'])
def proporcoes():
    predicoes_palavras = request.json
    predicoes_palavras = predicoes_palavras["dados"]

    # Mapear os sentimentos para os valores numéricos correspondentes
    mapeamento_sentimentos = {"negativo": 0, "neutro": 1, "positivo": 2}
    predicoes_numeros = [mapeamento_sentimentos[sentimento] for sentimento in predicoes_palavras]

    # Contar a ocorrência de cada sentimento
    proporcoes = Counter(predicoes_numeros)

    # Calcular as proporções
    total = len(predicoes_numeros)
    proporcoes = {sentimento: count/total for sentimento, count in proporcoes.items()}

    return json.dumps(proporcoes)

# Rota que retorna a nuvem de palavras 
@app.route('/nuvem-palavras', methods=['POST'])
def nuvem_palavras():
    dados = request.json["dados"]
    
    # Unir todos os textos em uma única string
    texto_completo = ' '.join(dados)

    # Criar a nuvem de palavras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_completo)

    # Plotar a nuvem de palavras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    # Salvar a imagem da nuvem de palavras em um arquivo
    imagem_nuvem = 'nuvem_palavras.png'
    plt.savefig(imagem_nuvem)

    # Retornar o nome do arquivo da imagem
    return jsonify({"imagem_nuvem": imagem_nuvem})


# Retorna o top 10 palavras
@app.route('/top-palavras', methods=['POST'])
def top_palavras():
    dados = request.json["dados"]

    # Unir todos os textos em uma única string
    texto_completo = ' '.join(dados)

    # Tokenizar o texto em palavras
    tokens = word_tokenize(texto_completo)

    # Remover stopwords das palavras tokenizadas
    stop_words = set(stop_words_custom )
    pontuacoes = set(string.punctuation)
    tokens_filtrados = [token for token in tokens if token.lower() not in stop_words and token.isalpha()]

    # Contar a ocorrência das palavras
    contagem_palavras = Counter(tokens_filtrados)

    # Obter as top 10 palavras mais frequentes
    top_palavras = contagem_palavras.most_common(10)

    # Retornar as top 10 palavras em formato JSON
    return jsonify({"top_palavras": top_palavras})

@app.route('/maiores-correlacoes', methods=['POST'])
def maiores_correlacoes():
    dados = request.json["dados"]

    # Aplicar a função emoji_to_word aos dados
    dados_processados = emoji_to_word(dados)

    # Aplicar a função processarTexto aos dados
    dados_processados = processarTexto(dados_processados)

    # Unir todos os textos em uma única lista de palavras
    palavras = [palavra for texto in dados_processados for palavra in texto if palavra != 'não']

    # Contar a frequência das palavras
    contagem_palavras = {}
    for palavra in palavras:
        if palavra in contagem_palavras:
            contagem_palavras[palavra] += 1
        else:
            contagem_palavras[palavra] = 1

    # Encontrar as palavras que mais aparecem na entrada
    palavras_mais_frequentes = sorted(contagem_palavras.items(), key=lambda x: x[1], reverse=True)[:20]

    # Criar um dicionário para armazenar as correlações das palavras mais frequentes
    maiores_correlacoes = {}

    # Encontrar as duas palavras que mais aparecem junto a cada palavra mais frequente
    for palavra, _ in palavras_mais_frequentes:
        palavras_relacionadas = [palavra]
        palavras_relacionadas.extend(
            sorted([p for p in palavras if p != palavra and p in contagem_palavras],
                   key=lambda p: contagem_palavras[p],
                   reverse=True)[:2]
        )
        maiores_correlacoes[palavra] = palavras_relacionadas

    # Retornar as maiores correlações em formato JSON
    return jsonify({"maiores_correlacoes": maiores_correlacoes})


@app.route('/ping', methods=['GET'])
def test():
    return ("Pong")


if __name__ == '__main__':
    app.run()
