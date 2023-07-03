from funcoes import pegaCaminhoArquivos


labels = ['A', 'B', 'C']

pasta_principal = 'letras'

caminho_dic = pegaCaminhoArquivos(pasta_principal)

a = caminho_dic['A']

data_treino = []

for arquivo  in a :
    with open(arquivo, 'r') as arquivo_aberto:
        conteudo = arquivo_aberto.read()
        data_treino.append(conteudo)

for e in data_treino:
    print(len(e))


