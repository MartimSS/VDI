# Dashboard de Análise de Andebol

Dashboard interativo desenvolvido em Streamlit para análise de performance em andebol, incluindo métricas de guarda-redes, equipa e tracking de jogadores.

## 📋 Requisitos

- Python 3.8 ou superior
- pip

## 🚀 Instalação e Execução

### 1. Clonar o repositório

```bash
git clone https://github.com/MartimSS/VDI.git
cd VDI
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Executar o dashboard

```bash
streamlit run dashboard7.py
```

O dashboard abrirá automaticamente no navegador em `http://localhost:8501`

## 📁 Estrutura de Ficheiros

```
VDI/
├── dashboard7.py          # Dashboard principal
├── datasets/              # Dados necessários
│   ├── goalkeeper.csv     # Dados dos guarda-redes
│   ├── shots.csv          # Dados dos remates
│   └── tracking.csv       # Dados de tracking
├── requirements.txt       # Dependências Python
└── README.md             # Este ficheiro
```

## 📊 Funcionalidades

### Aba GUARDA-REDES
- Indicadores principais (defesas, golos sofridos, tempo de reação)
- Heatmap da baliza (zonas onde sofre golos)
- Tempo de reação vs cansaço ao longo do jogo
- Evolução acumulada dos resultados

### Aba EQUIPA
- Remates sofridos e eficácia adversária
- Mapa de campo com zonas de remate
- Top jogadores adversários perigosos
- Ridgeline de velocidade instantânea
- Distância percorrida por jogador

### Aba CONCLUSÕES
- KPIs resumo (taxa de defesa, reação, eficácia adversária, intensidade)
- Evolução por período (match/session)
- Tendências ao longo do tempo
- Observações automáticas sobre melhorias

## 🎯 Filtros Disponíveis

- **Contexto**: Jogo ou Treino
- **Período**: Todos ou específico (match1, match2, session1, etc.)
- **Guarda-Redes**: Seleção múltipla de guarda-redes

## 🛠️ Tecnologias

- **Streamlit**: Framework de dashboard
- **Plotly**: Visualizações interativas
- **Pandas**: Manipulação de dados
- **NumPy**: Cálculos numéricos

## 📝 Notas

- Os dados devem estar na pasta `datasets/` no formato CSV
- Os ficheiros CSV devem seguir a estrutura especificada no código
- Para produção, considere configurar as variáveis de ambiente apropriadas

## 🔧 Deploy (Opcional)

Para deploy em Streamlit Cloud:

1. Fazer push do código para o GitHub
2. Aceder a [share.streamlit.io](https://share.streamlit.io)
3. Conectar o repositório GitHub
4. Selecionar o ficheiro `dashboard7.py`
5. Deploy automático!

---

**Dashboard desenvolvido para análise de performance em andebol**
