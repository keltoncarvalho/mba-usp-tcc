#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:37:29 2026

@author: Kelton Carvalho Andrade
"""
import json
import pandas as pd
import numpy as np
import pnadium
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

#%% Seção de download (Executar apenas se necessário baixar os dados novamente)
#path = './microdados'
#pnadium.anual.download(2024, 1, 'v', path, save_file=True) 
#%% Lendo Dataframe e Dicionário com variáveis de interesse da PNADc
caminho_vars = './dicionario.json'
with open(caminho_vars, 'r') as f:
    data = json.load(f)
variaveis = list(data.keys())
dicionario = pd.read_json(caminho_vars, encoding='UTF-8', orient='index')
df = pd.read_parquet('./microdados/pnad_anual_visita_012024.parquet', columns=variaveis)
# dropando pensionistas, empregados e filhos de empregados
df = df[~df["V2005"].isin([17, 18, 19])]
#%% Tratamento
# ===============================================================================================================
# 1. Dimensão TRABALHO
# ===============================================================================================================
# --- INDICADOR 1: INFORMALIDADE ---
# Miores de 18 trabalhando informalmente, por conta própria ou empregador, que não contribuem para previdência
informais = ((df['VD4009'].isin([2,4,6,8,9,10])) & (df['V2009'] >= 18) & (df['VD4012'] == 2))
# Maiores de 18 com trabalho formal, mas subocupadas por horas trabalhadas
subocupados = ((df['VD4009'].isin([1,3,5,7])) & (df['V2009'] >= 18) & (df['VD4004A'] == 1))
# União Subocupação + Informalidade
df['informal'] = np.where((informais | subocupados), 1, 0)
# Contágio para o domicílio
# A família é PRIVADA (1) se há pelo menos uma pessoa >= 18 anos em estado de informalidade ou subocupação.
df['informal_dom'] = df.groupby('COD_FAM')['informal'].transform('max')

# --- INDICADOR 2: DESEMPREGO ---
# Maiores de 18 desocupados
desocupados = ((df['VD4002'] == 2) & (df['V2009'] >= 18))
# Desocupação individual
df['desocupado'] = np.where(desocupados, 1, 0)
# Contágio para domicílio
df['desocupacao_dom'] = df.groupby('COD_FAM')['desocupado'].transform('max')
# Maiores de 18 desalentados
desalento = ((df['VD4005'] == 1) & (df['V2009'] >= 18))
# Desalento individual
df['desalentado'] = np.where(desalento, 1, 0)
# Contágio para domicílio
df['desalento_dom'] = df.groupby('COD_FAM')['desalentado'].transform('max')
# Privadoção no indicador (1) se há alguém desocupado ou desalentado no domicílio
df['pv_emprego'] = np.where((df['desocupado'] | df['desalentado']), 1, 0)
#%%
# ===============================================================================================================
# 2. Dimensão PADRÃO DE VIDA
# ===============================================================================================================
# --- INDICADOR 1: MATERIAL DO DOMICÍLIO ---
# Parede inadequada
parede = (df['S01002'].isin([3, 5, 6]))
# Cobertura inadequada 
cobertura = (df['S01003'] == 6) 
# Piso inadequado
piso = (df['S01004'].isin([4,5])) 
# Privação Final Mat. Domicílio
df['pv_mat_dom']= np.where((parede | cobertura | piso), 1, 0)

# --- INDICADOR 2: ACESSO À ÁGUA ---
# Privação por fonte inadequada
fonte = (df['S01007'].isin([3, 4, 5, 6]))
# Privação por fonte adequada, mas de abastecimento intermitente e sem reservatório declarado
frequencia = ((df['S01007'] == 1) & (df['S01008'] != 1) & (df['S01009'] != 1))
# Privação domiciliar de acessoa à água
df['pv_agua'] = np.where((fonte | frequencia), 1, 0)

# --- INDICADOR 3: SANEAMENTO BÁSICO ---
# Privação por falta de banheiro exclusivo
banheiro = (df['S01011A'] == 0)
# Privação por escoamento inadequado
escoamento = (~df['S01012A'].isin([1,2,3]))
# Privação por destino inadequado do lixo
lixo = (df['S01013'].isin([3,4,5,6]))
# Privação domiciliar de Saneamento Básico
df['pv_san_bas'] = np.where((banheiro | escoamento | lixo), 1, 0)

# --- INDICADOR 4: ENERGIA ---
# ELETRICIDADE
# Privação por não usar energia elétrica
nao_usa = (df['S01014'] != 1)
# Privação por ineficiência da rede e ausência de fonte alternativa
ineficiente = ((df['S01014'] == 1) & (df['S010141'] == 1) & (df['S01015'] != 1) & (df['S010142'] != 1))
# Privação de eletricidade por qualquer motivo
df['pv_elet'] = np.where((nao_usa | ineficiente), 1, 0)

# COMBUSTÍVEL 
# Privação por não usar combustível para preparar alimentos (indica que não prepara)
nao_usa = (df['S01016A'] == 2)
# Usa combustível inadequado
inadequado = ((df['S01016A3'] == 1) | (df['S01016A5'] == 1)) 
# Não usa combustível adequado/limpo
nao_limpo = ((df['S01016A1'] == 2) & (df['S01016A2'] == 2) & (df['S01016A4'] == 2))
# Privação (1): O não uso de combustível adequado indica que o combustível inadequado é o principal utilizado
# para preparar alimentos
df['pv_comb'] = np.where((nao_usa | (inadequado & nao_limpo)), 1, 0)

# Privação agregada de energia (combustível ou eletricidade):
agregado = ((df['pv_elet'] + df['pv_comb']) > 0)
df['pv_energ'] = np.where(agregado, 1, 0)

# --- INDICADOR 5: GELADEIRA ---
# Privado se não possui geladeira
df['pv_gld'] = np.where((df['S01023'] == 3), 1, 0) 

# --- INDICADOR 6: SEGURANÇA DA POSSE (PROPRIEDADE) ---
# Valor do Salário Mínimo no ano de referência
sal_min_24 = 1412.00
# Imóvel cedido por parente ou cedido de outra forma ou ocupado
ced_oc = (df['S01017'].isin([5, 6, 7]))
# Peso excessivo do alugel: gasto acima de 30% para rendas abaixo de 3 Salários Mínimos
aluguel = ((df['S01017'] == 3) & (((df['S01019']/df['VD5007']) > 0.30) & (df['VD5007'] < (3*sal_min_24))))
# Privação por insegurança da posse
df['pv_propriedade'] = np.where((ced_oc | aluguel), 1, 0)
#%% 
# ===============================================================================================================
# 3. Dimensão EDUCAÇÃO
# ===============================================================================================================
# --- INDICADOR 1: FREQUÊNCIA ESCOLAR (Contágio de privação) ---
# Não estudam, mas estão na faixa de idade obrigatória (4 até 17)
fora_escola = ((df['V3002'] == 2) & ((df['V2009'] >= 4) & (df['V2009'] <= 17)))
# Coluna de privação individual em frequência escolar
df['fq_esc_in']  = np.where(fora_escola, 1, 0)
# Privação familiar: contágio
df['pv_fq_esc'] = df.groupby('COD_FAM')['fq_esc_in'].transform('max')

# --- INDICADOR 2: ATRASO ESCOLAR (Contágio de Privação) ---
# * Ensino Fundamental Regular (Atraso de 2 anos ou mais)
cond_fund = ((df['V3003A'] == 4) & ((df['V2009'] - (df['V3006'] + 5)) >= 2))
# * Ensino Médio Regular - Divisão por Séries de 1 a 3
cond_medio_serie = ((df['V3003A'] == 6) & (df['V3006'] <= 4) & ((df['V2009'] - (df['V3006'] + 14)) >= 2))
# Divisão por Anos de 10 a 12
cond_medio_ano = ((df['V3003A'] == 6) & (df['V3006'].isin([10, 11, 12])) & ((df['V2009'] - (df['V3006'] + 5)) >= 2))
# * EJA (Educação de Jovens e Adultos)
# Se tem 17 anos ou menos e está no EJA, tem defasagem escolar.
# A idade mínima para ter 2 anos de atraso é a partir dos 8 anos.
cond_eja = (df['V3003A'].isin([5, 7])) & (df['V2009'] <= 17) & (df['V2009'] >= 8)
# 1. Se cair em qualquer uma das condições acima, a criança está atrasada (1)
df['ats_esc_in'] = np.where((cond_fund | cond_medio_serie | cond_medio_ano | cond_eja), 1, 0)
# 2. Contágio para o domicílio
df['pv_ats_esc'] = df.groupby('COD_FAM')['ats_esc_in'].transform('max')

# --- INDICADOR 3: MAIOR FORMAÇÃO (Contágio de Proteção) ---
# 1. Identificar um "Adulto Protetor" (>18 E tem Ensino Médio Completo ou superior)
protecao = ((df['V2009'] >= 18) & (df['VD3004'] >= 5))
df['adt_prot'] = np.where(protecao, 1, 0)
# 2. Total adultos protegidos na família
df['qtd_prot'] = df.groupby('COD_FAM')['adt_prot'].transform('sum')
# 3. Privação: se tem 0 "adultos protetores" na família, é privado no indicador de formação
df['pv_form'] = np.where((df['qtd_prot'] == 0), 1, 0)
#%% 
# ===============================================================================================================
# 4. Dimensão TICs
# ===============================================================================================================
# --- INDICADOR 1: ACESSO À INTERNET ---
# Não acessa internet
df['internet'] = np.where((df['S01029'] == 2), 1, 0)
# Contágio de proteção (Uma pessoa com acesso protege (0) o domicílio da privação (1))
df['pv_net'] = df.groupby('COD_FAM')['internet'].transform('min')

# --- INDICADOR 2: EQUIPAMENTOS ---
# Não possui equipamentos de TICs.
sem_equipamentos = ((df['S01021'] == 0) & (df['S01028'] == 2))
df['sem_equipamentos'] = np.where(sem_equipamentos, 1, 0)
# Contágio de proteção
df['pv_equip'] = df.groupby('COD_FAM')['sem_equipamentos'].transform('min')
#%% Pobreza Monetária e Auxílios (tratamento)
# Definindo a Pobreza Monetária (2024: R$218,00)
linha_pobreza = 218
# VD5008 é a Renda Domiciliar Per Capita 
df['pobre_monetario'] = np.where((df['VD5008'] <= linha_pobreza), 1, 0)
# Contágio de auxílio (nem sempre o responsável pelo domicílio é o titular)
# Na PNAD recebe = 1 e não_recebe = 2, logo transform('min')
df['recebe_pbf'] = df.groupby('COD_FAM')['V5002A'].transform('min')

#%% 
# ===============================================================================================================
# Cenário de Pesos e Linha de Corte (Ref: IPM-Global)
# ===============================================================================================================
pesos_ipm = {
    # DIMENSÃO TRABALHO (Total: 0.25)
    'informal_dom': 0.125,
    'pv_emprego': 0.125,

    # DIMENSÃO PADRÃO DE VIDA (Total: 0.25)
    'pv_mat_dom': 0.0417,
    'pv_agua': 0.0417,
    'pv_san_bas': 0.0417,
    'pv_energ': 0.0417,
    'pv_gld': 0.0417,
    'pv_propriedade': 0.0417,

    # DIMENSÃO EDUCAÇÃO (Total: 0.25)
    'pv_fq_esc': 0.0833,
    'pv_ats_esc': 0.0833,
    'pv_form': 0.0833,

    # DIMENSÃO TICs (Total: 0.25)
    'pv_net': 0.125,
    'pv_equip': 0.125
}

pesos_definitivos = pd.Series(pesos_ipm)
# Corte de 33.33%
k_definitivo = 0.3333 
# 2. Cálculo do escore final
df['pv_score'] = df[pesos_ipm.keys()].mul(pesos_definitivos).sum(axis=1)
# 3. Identificação: é pobre (1) e não é pobre (0)
df['pobre_multidimensional'] = np.where((df['pv_score'] >= k_definitivo), 1, 0)
# 4. Censura: Zera as privações de quem não atingiu o corte k
df['score_censurado_final'] = df['pv_score'] * df['pobre_multidimensional']
#%%
# ===============================================================================================================
# INTENSIDADE E IPM (PBF X SEM PBF)
# ===============================================================================================================
# 1. ISOLANDO OS GRUPOS
df_pobres = df[df['pobre_multidimensional'] == 1]
df_pbf = df[df['recebe_pbf'] == 1]
pobres_pbf = df_pbf[df_pbf['pobre_multidimensional'] == 1]
df_sempbf = df[df['recebe_pbf'] == 2]
pobres_sempbf = df_sempbf[df_sempbf['pobre_multidimensional'] == 1]

# 2. CALCULANDO AS MEDIDAS AF
# --- INCIDÊNCIA H ---
H_pobres = np.average(df['pobre_multidimensional'], weights=df['V1032'])
H_pbf = np.average(df_pbf['pobre_multidimensional'], weights=df_pbf['V1032'])
H_sempbf =  np.average(df_sempbf['pobre_multidimensional'], weights=df_sempbf['V1032'])
# --- INTENSIDADE A ---
A_pobres = np.average(df_pobres['pv_score'], weights=df_pobres['V1032'])
A_pbf = np.average(pobres_pbf['pv_score'], weights=pobres_pbf['V1032'])
A_sempbf = np.average(pobres_sempbf['pv_score'], weights=pobres_sempbf['V1032'])
# --- CÁLCULO DE M0 (IPM Ajustado / Adjusted Headcount Ratio) ---
M0_pobres = H_pobres * A_pobres
M0_pbf = H_pbf * A_pbf
M0_sempbf = H_sempbf * A_sempbf

print("--- RESULTADOS OFICIAIS DO ÍNDICE ALKIRE-FOSTER ---")
print(f"H (Incidência):  {H_pobres * 100:.2f}% (Proporção de pessoas POBRES Nacionalmente)")
print(f"A (Intensidade): {A_pobres * 100:.2f}% (Média de privações que os POBRES sofrem)")
print(f"M0 (Índice IPM): {M0_pobres:.4f} (IPM-PBF)")
print('---------------------------------------------------------------------------------')
print(f"H (Incidência):  {H_pbf * 100:.2f}% (Proporção de pessoas POBRES no PBF)")
print(f"A (Intensidade): {A_pbf * 100:.2f}% (Média de privações que os POBRES do PBF sofrem)")
print(f"M0 (Índice IPM): {M0_pbf:.4f} (IPM-PBF)")
print('---------------------------------------------------------------------------------')
print(f"H (Incidência):  {H_sempbf * 100:.2f}% (Proporção de pessoas POBRES sem PBF)")
print(f"A (Intensidade): {A_sempbf * 100:.2f}% (Média de privações que os POBRES sem PBF sofrem)")
print(f"M0 (Índice IPM): {M0_sempbf:.4f} (IPM-SEM PBF)")
#%% COBERTURA DO PBF
# ===============================================================================================================
# VERIFICANDO O PONTO CEGO DO PROGRAMA
# ===============================================================================================================
# Ponto Cego: É pobre IPM E não tem PBF E é pobre monetário
ponto_cego = (df['pobre_multidimensional'] == 1) & (df['recebe_pbf'] == 2) & (df['pobre_monetario'])
# Expansão para população do Brasil 
populacao_ponto_cego = df.loc[ponto_cego, 'V1032'].sum()
total_pobres_ipm = df.loc[df['pobre_multidimensional'] == 1, 'V1032'].sum()
total_pobres_pbf = df_pbf.loc[df_pbf['pobre_multidimensional'] == 1, 'V1032'].sum()
total_pobres_sempbf = df_sempbf.loc[df_sempbf['pobre_multidimensional'] == 1, 'V1032'].sum()
# Taxa de Exclusão
taxa_exclusao = (populacao_ponto_cego / total_pobres_ipm) * 100
print(f"Total de pobres pelo IPM: {total_pobres_ipm:,.0f} brasileiros")
print(f"Pobres IPM SEM Bolsa Família (Ponto Cego): {populacao_ponto_cego:,.0f} brasileiros")
print(f"Taxa de Exclusão: {taxa_exclusao:.2f}% dos pobres multidimensionais estão fora do PBF.")
#%% 
# ===============================================================================================================
# PLOTAGEM DO PONTO CEGO
# ===============================================================================================================
# 1. ISOLANDO O "PONTO CEGO"
# Filtro: pobres multidimensionais e NÃO recebem PBF
df_ponto_cego = df[(df['pobre_multidimensional'] == 1) & (df['recebe_pbf'] == 2)]

# 2. CÁLCULO DOS VOLUMES POPULACIONAIS (3 FAIXAS DE RENDA)
extrema_pob = df_ponto_cego[df_ponto_cego['VD5008'] <= 218]['V1032'].sum()
vulneraveis = df_ponto_cego[(df_ponto_cego['VD5008'] > 218) & (df_ponto_cego['VD5008'] <= 706)]['V1032'].sum()
apenas_multi = df_ponto_cego[df_ponto_cego['VD5008'] > 706]['V1032'].sum()

total_ponto_cego = round(extrema_pob + vulneraveis + apenas_multi)

# Calculando as taxas relativas
pct_extrema = (extrema_pob / total_ponto_cego) * 100
pct_vuln = (vulneraveis / total_ponto_cego) * 100
pct_multi = (apenas_multi / total_ponto_cego) * 100

# 3. PLOTAGEM DO GRÁFICO (DONUT CHART TRIPLO)
fig, ax = plt.subplots(figsize=(10, 7))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Liberation Sans']

tamanhos = [extrema_pob, vulneraveis, apenas_multi]
# Cores: Vermelho (Urgência), Laranja (Atenção), Cinza (Estrutural)
cores = ['#d62728', '#ff7f0e', '#7f7f7f'] 

# Rótulos da legenda
labels = [
    'Extrema Pobreza Monetária\n(Renda <= R\$ 218,00)',
    'Vulneráveis de Baixa Renda\n(R\$ 219,00 a R\$ 706,00)',
    'Pobreza Apenas Multidimensional\n(Renda > R\$ 706,00)'
]

# função para cálculo e formatação dos valores dentro das fatias
def formato_duplo(pct, todos_valores):
    # Calcula o valor absoluto reverso a partir da porcentagem
    absoluto = (pct / 100.0) * sum(todos_valores)
    # Retorna "X.X M \n (YY.Y%)"
    return f"{absoluto/1e6:.1f} M\n({pct:.1f}%)"

wedges, texts, autotexts = ax.pie(
    tamanhos, 
    colors=cores,
    startangle=90,
    # Passagem da função usando lambda
    autopct= lambda pct: formato_duplo(pct, tamanhos),       
    pctdistance=0.8,         
    textprops=dict(color="black", fontsize=11), 
    wedgeprops=dict(width=0.4) 
)


# Legenda lateral
plt.legend(
    wedges, 
    labels, 
    title="Perfil do Ponto Cego\n(Não Beneficiários):", 
    loc="center left", 
    bbox_to_anchor=(0.95, 0.5), 
    fontsize=11, 
    title_fontsize=11,
    frameon=False
)

plt.title("Decomposição do 'Ponto Cego' Multidimensional\npor Faixas Oficiais de Renda e Vulnerabilidade", 
          fontsize=14, fontweight='bold', pad=10)

# Texto central (todo o grupo arredondado uma casa decimal)
ax.text(0, 0, f"Total do\nGrupo:\n{total_ponto_cego/1e6:.1f} M", 
        ha='center', va='center', fontsize=11)

plt.tight_layout()
#plt.savefig('decomposicao_ponto_cego_texto_duplo.tiff', dpi=300, bbox_inches='tight')
#plt.savefig('decomposicao_ponto_cego_texto_duplo.jpeg', dpi=300, bbox_inches='tight')
plt.show()
#%%
# ===============================================================================================================
# DECOMPOSIÇÃO POR CONTRIBUIÇÃO DE CADA DIMENSÃO
# ===============================================================================================================
# --- 1. CÁLCULO DAS CONTRIBUIÇÕES INDIVIDUAIS COM OS PESOS ---
df['nome_grupo'] = np.where(df['recebe_pbf'] == 1 , 'Com Auxílio', 'Sem Auxílio')
for ind, peso in pesos_definitivos.items():
    privacao_censurada = df[ind] * df['pobre_multidimensional']
    df[f'contrib_{ind}'] = privacao_censurada * df['V1032'] * peso

# --- 2. AGRUPAMENTO POR DIMENSÕES ---
df['dim_Trabalho'] = df['contrib_informal_dom'] + df['contrib_pv_emprego']

df['dim_Padrão de Vida'] = (df['contrib_pv_mat_dom'] + df['contrib_pv_agua'] + 
                            df['contrib_pv_san_bas'] + df['contrib_pv_energ'] + 
                            df['contrib_pv_gld'] + df['contrib_pv_propriedade'])

df['dim_Educação'] = df['contrib_pv_fq_esc'] + df['contrib_pv_ats_esc'] + df['contrib_pv_form']

df['dim_TICs'] = df['contrib_pv_net'] + df['contrib_pv_equip']

colunas_dimensoes = ['dim_Trabalho', 'dim_Padrão de Vida', 'dim_Educação', 'dim_TICs']

# --- 3. AGREGAÇÃO POR GRUPOS E CÁLCULOS RELATIVOS ---
df_agregado = df.groupby('nome_grupo')[colunas_dimensoes].sum()
populacao_por_grupo = df.groupby('nome_grupo')['V1032'].sum()

ipm_abs_por_grupo = df_agregado.div(populacao_por_grupo, axis=0)
ipm_rel_por_grupo = ipm_abs_por_grupo.div(ipm_abs_por_grupo.sum(axis=1), axis=0) * 100

ordem_desejada = ['Com Auxílio', 'Sem Auxílio']
ipm_rel_por_grupo = ipm_rel_por_grupo.reindex(ordem_desejada)

populacao_total = df['V1032'].sum()
contrib_total = df[colunas_dimensoes].sum()

ipm_abs_total = contrib_total / populacao_total
ipm_rel_total = (ipm_abs_total / ipm_abs_total.sum()) * 100

df_nacional = pd.DataFrame([ipm_rel_total.values], columns=ipm_rel_total.index, index=['Nacional'])
df_final = pd.concat([df_nacional, ipm_rel_por_grupo])

df_final.columns = [col.replace('dim_', '') for col in df_final.columns]

# --- 4. PLOTAGEM DO GRÁFICO ---
fig, ax = plt.subplots(figsize=(10, 8))

cores_viridis = ['#414487', '#31688e', '#35b779', '#fde725']

df_final.plot(
    kind='bar', 
    stacked=True, 
    ax=ax,
    color=cores_viridis, 
    width=0.5,
    zorder=3
)

#plt.title('Composição do IPM por Dimensões\n(Nacional vs. Auxílio)', fontsize=15, pad=20, fontweight='bold')
plt.ylabel('Contribuição Relativa para o IPM (%)', fontsize=11)
plt.xlabel('')
plt.xticks(rotation=0, fontsize=11) 
plt.grid(False)

for container in ax.containers:
    for barra in container:
        altura = barra.get_height()
        if altura > 0:
            x = barra.get_x() + barra.get_width() / 2
            y = barra.get_y() + altura / 2
            
            ax.text(x, y, f'{altura:.1f}%', ha='center', va='center', 
                    color='black', fontsize=11)

plt.legend(
    title='Dimensões', 
    loc='upper left',
    bbox_to_anchor=(1, 1),
    ncol=1,
    frameon=False, 
    fontsize=11,
    title_fontsize=11
)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.5)

plt.tight_layout()
#plt.savefig('composicao_ipm_dimensoes_lateral.tiff', dpi=300, bbox_inches='tight')
#plt.savefig('composicao_ipm_dimensoes_lateral.jpeg', dpi=1200, bbox_inches='tight')
plt.show()

#%% 
# ===============================================================================================================
# DECOMPOSIÇÃO POR FREQUÊNCIA DE PRIVAÇÕES POR GRUPOS
# ===============================================================================================================
# Beneficiário (1) e Não Beneficiário (2)
df['nome_grupo'] = np.where(df['recebe_pbf'] == 1, 'Com Auxílio', 'Sem Auxílio')

# --- 1. CÁLCULO DA FREQUÊNCIA RELATIVA (TAXA DE PRIVAÇÃO CENSURADA) ---
for ind in pesos_definitivos.keys():
    df[f'freq_{ind}'] = df[ind] * df['pobre_multidimensional'] * df['V1032']

colunas_freq = [f'freq_{ind}' for ind in pesos_definitivos.keys()]

# Agrupando e somando as pessoas com privação usando o grupo
df_agregado_freq = df.groupby('nome_grupo')[colunas_freq].sum()
populacao_por_grupo = df.groupby('nome_grupo')['V1032'].sum()
# Normalização
taxas_por_grupo = df_agregado_freq.div(populacao_por_grupo, axis=0) * 100
# Ordena o índice com apenas 2 grupos
ordem_desejada = ['Com Auxílio', 'Sem Auxílio']
taxas_por_grupo = taxas_por_grupo.reindex(ordem_desejada)

# --- 2. CÁLCULO DA TAXA NACIONAL (LINHA DE BASE) ---

aliases_legenda = {
    'informal_dom': 'Informalidade/\n Subocupação',
    'pv_emprego': 'Desemprego/\n Desalento',
    'pv_mat_dom': 'Material do\n Domicílio',
    'pv_agua': 'Água',
    'pv_san_bas': 'Saneamento\n Básico',
    'pv_energ': 'Energia',
    'pv_gld': 'Geladeira',
    'pv_propriedade': 'Propriedade',
    'pv_fq_esc': 'Frequência\n Escolar',
    'pv_ats_esc': 'Atraso\n Escolar',
    'pv_form': 'Ensino Médio',
    'pv_net': 'Acesso à\n Internet',
    'pv_equip': 'Equipamentos\n de TICs'
}

populacao_total = df['V1032'].sum()
freq_total = df[colunas_freq].sum()

taxas_nacional = (freq_total / populacao_total) * 100
df_nacional_freq = pd.DataFrame([taxas_nacional.values], columns=taxas_nacional.index, index=['Nacional'])

# Juntando Nacional com os 2 Subgrupos
df_freq_final = pd.concat([df_nacional_freq, taxas_por_grupo])

# Nomeando as colunas com base no dicionário
df_freq_final.columns = [col.replace('freq_', '') for col in df_freq_final.columns]
df_freq_final = df_freq_final.rename(columns=aliases_legenda)

# --- 3. TRANSPOSIÇÃO DOS DADOS ---
# Transpõe a matriz: Indicadores nas linhas (Eixo X), Grupos nas colunas (Legenda)
df_grafico = df_freq_final.T 
# Ordenando os indicadores do maior para o menor (Nacional como referência)
df_grafico = df_grafico.sort_values(by='Nacional', ascending=True)
# TAMANHO DA FIGURA (Maior altura)
fig, ax = plt.subplots(figsize=(10, 12)) 

df_grafico.plot(
    kind='barh', 
    ax=ax, 
    width=0.8,
    colormap='viridis',
    zorder=3
)

# Customização 
#plt.title('Taxa de Privação Censurada por Indicador\n(Nacional vs. Auxílio)', \
          #fontsize=15, pad=15, fontweight='bold')
plt.xlabel('Frequência na População do Grupo (%)', fontsize=11)
plt.ylabel('')
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)
plt.grid(False)

# --- 4. AJUSTE DOS RÓTULOS DE DADOS PARA A HORIZONTAL ---
for container in ax.containers:
    for barra in container:
        largura = barra.get_width()
        if largura > 1: # Mostra se for maior que 1%
            ax.annotate(f'{largura:.1f}%',
                        xy=(largura, barra.get_y() + barra.get_height() / 2),
                        xytext=(4, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=11)

# Movendo a legenda para fora do gráfico
plt.legend(
    title='Grupos de Análise', 
    fontsize=11, 
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.05), # Centralizando no eixo X, recuo de 0.05 no Y
    ncol=3, # Divisão em 3 colunas
    frameon=False
)

plt.tight_layout()
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.5)
#plt.savefig('grafico_privacao.tiff', dpi=300, bbox_inches='tight')
#plt.savefig('grafico_privacao.jpeg', dpi=300, bbox_inches='tight')
plt.show()
#%%
# ===============================================================================================================
# LOGIT COM PESOS NORMALIZADOS
# ===============================================================================================================
# agrupando por região para reduzir dimensionalidade
df['zona'] = np.where((df['V1022'] == 1), 1, 0)
# tratamento de outras variáveis: o modelo "conta" a partir do 0
df['sexo'] = np.where((df['V2007'] == 1), 1, 0)
df['cor'] = np.where((df['V2010'] == 1), 0, (df['V2010'] - 1))
df['recebe_pbf'] = np.where((df['recebe_pbf'] == 1), 1, 0)
df['escolaridade'] = np.where((df['VD3004'] == 1), 0, (df['VD3004'] - 1))
# Mapeamento de regiões pela UF
mapa_regioes = {
    1: 'Norte',
    2: 'Nordeste',
    3: 'Sudeste',
    4: 'Sul',
    5: 'Centro-Oeste'
}
df['regiao'] = (df['UF'] // 10).map(mapa_regioes)
df['regiao'] = pd.Categorical(df['regiao'], categories=['Norte', 'Nordeste', 'Sudeste', 'Sul', 'Centro-Oeste'],\
                              ordered=True)
# filtrando os responsáveis pelos domicílios para evitar peso por repetição/dependência
df_responsaveis = df[df['V2005'] == 1].copy()
# normalizando pesos para usar no modelo
df_responsaveis['peso_normalizado'] = df_responsaveis['V1032'] / \
    df_responsaveis['V1032'].sum() * len(df_responsaveis)

# instanciando o modelo glm
modelo_df_responsaveis = smf.glm(
    # fórmula para reduzir dimensionalidade e variáveis tratadas
    formula='pobre_multidimensional ~  C(regiao, Treatment(reference="Sudeste")) + C(zona) + C(sexo) + \
        V2009 + I(V2009 **2) + C(cor) + VD2003 + C(recebe_pbf) + C(adt_prot) + np.log1p(VD5008)',
    data=df_responsaveis, 
    family=sm.families.Binomial(link=sm.families.links.Logit()),
    freq_weights=df_responsaveis['peso_normalizado']
).fit(cov_type='HC1')

print(modelo_df_responsaveis.summary())
#%% Sumário resumido
summary_col([modelo_df_responsaveis],
            model_names=["MODELO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-likelihood':lambda x: "{:.3f}".format(x.llf)
        })

#%% Forest Plot
# ===============================================================================================================
# FOREST PLOT (GLM)
# ===============================================================================================================
# --- 1. EXTRAÇÃO DOS DADOS DO MODELO ---
# Extraindo os coeficientes e os limites inferior e superior (IC 95%)
df_forest = pd.DataFrame({
    'coef': modelo_df_responsaveis.params,
    'lower': modelo_df_responsaveis.conf_int()[0],
    'upper': modelo_df_responsaveis.conf_int()[1],
    'pvalue': modelo_df_responsaveis.pvalues
})

# Removendo o Intercepto, pois a chance base não entra no gráfico
if 'Intercept' in df_forest.index:
    df_forest = df_forest.drop('Intercept')
    
# --- 2. CONVERSÃO PARA ODDS RATIO ---
# Aplicando a função exponencial (np.exp)
df_forest['OR'] = np.exp(df_forest['coef'])
df_forest['OR_lower'] = np.exp(df_forest['lower'])
df_forest['OR_upper'] = np.exp(df_forest['upper'])

# --- 3. LIMPEZA DOS NOMES ---
dicionario_nomes = {
    'C(regiao, Treatment(reference="Sudeste"))[T.Centro-Oeste]': "Centro-Oeste",
    'C(regiao, Treatment(reference="Sudeste"))[T.Norte]': "Norte",
    'C(regiao, Treatment(reference="Sudeste"))[T.Nordeste]': "Nordeste",
    'C(regiao, Treatment(reference="Sudeste"))[T.Sul]' : "Sul",
    "C(recebe_pbf)[T.1]": "Recebe Bolsa Família",
    "C(V1022)[T.2]": "Situação Rural",
    "np.log1p(VD5008)": "Renda Per Capita (Log)",
    "C(zona)[T.1]" : "Urbana",
    "C(sexo)[T.1]" : "Homem",
    "C(cor)[T.1]" : "Preta",
    "C(cor)[T.2]" : "Amarela",
    "C(cor)[T.3]" : "Parda",
    "C(cor)[T.4]" : "Indígena",
    "C(adt_prot)[T.1]" : "Ensino Médio Completo",
    "V2009" : "Idade",
    "I(V2009 ** 2)" : "Idade²",
}

# Filtro: apenas variáveis no dict
df_forest = df_forest.loc[df_forest.index.intersection(dicionario_nomes.keys())].copy()
# Renomeia as linhas do index
df_forest = df_forest.rename(index=dicionario_nomes)

# Ordena do maior Odds Ratio para o menor
df_forest = df_forest.sort_values(by='OR', ascending=True)

# --- 4. PLOTAGEM DO FOREST PLOT ---
fig, ax = plt.subplots(figsize=(10, 8))

# Calculando distância do centro até os limites
erros_inferiores = df_forest['OR'] - df_forest['OR_lower']
erros_superiores = df_forest['OR_upper'] - df_forest['OR']
erros = [erros_inferiores, erros_superiores]

ax.errorbar(
    x=df_forest['OR'], 
    y=df_forest.index, 
    xerr=erros, 
    fmt='o', 
    color='#1f77b4', 
    ecolor='#31688e', 
    elinewidth=2.5,   
    capsize=4,        
    markersize=8      
)

# Referência de Efeito Nulo (OR = 1)
ax.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=0)

# Customização e estética
#plt.title('Determinantes da Pobreza Multidimensional\nRazão de Chances (Odds Ratio) e IC 95%', \
    #fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Odds Ratio (Menor que 1: Proteção | Maior que 1: Risco)', fontsize=11)
plt.yticks(fontsize=11)
plt.grid(False)

# Adiciona o número exato acima do ponto
for i, (or_val, p_val) in enumerate(zip(df_forest['OR'], df_forest['pvalue'])):
    # Adiciona asterisco se for estatisticamente significante
    significancia = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    texto = f"{or_val:.2f}{significancia}"
    ax.text(or_val, i + 0.25, texto, va='center', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.5)
#plt.savefig('forest_plot_determinantes.tiff', dpi=300, bbox_inches='tight')
#plt.savefig('forest_plot_determinantes.jpeg', dpi=300, bbox_inches='tight')
plt.show()
