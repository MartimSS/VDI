import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
import re
from pathlib import Path

# Função auxiliar para carregar imagem como base64
def get_image_base64(image_path):
    """Converte imagem local para base64 para usar no Plotly"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Configuração da página
st.set_page_config(
    page_title="Dashboard Análise Andebol",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Config padrão do Plotly (ex.: exportar PNG com upscale/maior resolução)
DEFAULT_PLOTLY_IMAGE_SCALE = 3
MAPA_X_ZOOM = 1 # <1 = mostra menos range em X (mais zoom/visibilidade)
MAPA_Y_ZOOM = 1# <1 = mostra menos range em Y (mais zoom/visibilidade)
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "scale": DEFAULT_PLOTLY_IMAGE_SCALE,
    },
}

# CSS customizado para layout compacto
st.markdown("""
    <style>
    /* Garantir que o header/toolbar do Streamlit fica visível */
    header[data-testid="stHeader"] {
        visibility: visible !important;
        display: block !important;
        top: 0.75rem !important;
        z-index: 9999 !important;
    }
    div[data-testid="stToolbar"] {
        visibility: visible !important;
        display: flex !important;
        z-index: 9999 !important;
    }
    #MainMenu {
        visibility: visible !important;
    }

    /* Reduzir padding geral */
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Reduzir tamanho de títulos */
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0.3rem !important;
        padding-bottom: 0 !important;
    }
    
    h2 {
        font-size: 1.3rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    h3 {
        font-size: 1.1rem !important;
        margin-top: 0.1rem !important;
        margin-bottom: 0.2rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Reduzir espaçamento entre elementos */
    .element-container {
        margin-bottom: 0.1rem !important;
    }
    
    /* Reduzir tamanho das métricas */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.65rem !important;
    }
    
    /* Reduzir espaçamento vertical das métricas */
    [data-testid="stMetric"] {
        padding: 0.3rem 0 !important;
    }
    
    [data-testid="stMetricLabel"] > div {
        margin-bottom: 0.1rem !important;
    }
    
    /* Reduzir espaço das tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding-top: 0.3rem;
        padding-bottom: 0.3rem;
    }
    
    /* Reduzir espaço acima das tabs */
    .stTabs {
        margin-top: 0rem !important;
    }
    
    /* Reduzir linhas divisórias */
    hr {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
    
    /* Sidebar mais compacto */
    .css-1d391kg, [data-testid="stSidebar"] {
        padding-top: 1rem;
    }
    
    /* Reduzir espaço dos gráficos */
    .js-plotly-plot {
        margin-bottom: 0 !important;
    }

    /* KPI Card (HTML/CSS, independente da versão do Streamlit) */
    .kpi-card {
        background-color: rgba(232, 245, 255, 0.92);
        border: 1px solid rgba(52, 152, 219, 0.55);
        border-left: 6px solid rgba(52, 152, 219, 0.95);
        border-radius: 12px;
        padding: 0.55rem 0.75rem;
        display: flex;
        gap: 0.75rem;
        align-items: stretch;
        justify-content: space-between;
        min-height: 76px;
    }
    .kpi-item {
        flex: 1;
        min-width: 0;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: rgba(49, 51, 63, 0.85);
        margin-bottom: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 650;
        color: rgba(49, 51, 63, 0.95);
        line-height: 1.15;
    }
    @media (max-width: 900px) {
        .kpi-card { flex-wrap: wrap; }
        .kpi-item { flex: 1 1 45%; }
    }
    </style>
    """, unsafe_allow_html=True)

# Carregar dados
@st.cache_data
def load_data():
    gkdf = pd.read_csv('datasets/goalkeeper.csv')
    shotsdf = pd.read_csv('datasets/shots.csv')
    trdf = pd.read_csv('datasets/tracking.csv')
    return gkdf, shotsdf, trdf

gkdf, shotsdf, trdf = load_data()


def _filter_ids_by_prefix(values, prefix: str):
    vals = [v for v in values if isinstance(v, str)]
    return sorted([v for v in vals if v.startswith(prefix)])


def filter_by_context_and_period(df: pd.DataFrame, context_filter: str, selected_period: str) -> pd.DataFrame:
    """Aplica filtro de Contexto (Jogo/Treino) e Período (matchX/sessionY).

    Suporta datasets que usam `match_id` e/ou `session_id` (ex: tracking).
    """
    if df is None or df.empty:
        return df

    prefix = 'match' if context_filter == 'Jogo' else 'session'

    # Escolher a coluna de ID mais provável por contexto
    if context_filter == 'Jogo':
        id_col = 'match_id' if 'match_id' in df.columns else ('session_id' if 'session_id' in df.columns else None)
    else:
        id_col = 'session_id' if 'session_id' in df.columns else ('match_id' if 'match_id' in df.columns else None)

    if id_col is None:
        return df

    out = df[df[id_col].astype(str).str.startswith(prefix)].copy()

    if selected_period and selected_period != 'Todos':
        selected_period_str = str(selected_period)
        mask = out[id_col].astype(str) == selected_period_str
        # Alguns datasets podem ter ambos os campos; aceitar qualquer um deles
        if 'match_id' in out.columns:
            mask = mask | (out['match_id'].astype(str) == selected_period_str)
        if 'session_id' in out.columns:
            mask = mask | (out['session_id'].astype(str) == selected_period_str)
        out = out[mask].copy()

    return out

# Funções auxiliares
def time_to_seconds(time_str):
    try:
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return None

def time_to_minutes(time_str):
    try:
        parts = time_str.split(':')
        return int(parts[0]) + int(parts[1]) / 60
    except:
        return None


def _tracking_period_col(df: pd.DataFrame, context_filter: str) -> str | None:
    if df is None or df.empty:
        return None
    if context_filter == 'Jogo':
        if 'match_id' in df.columns:
            return 'match_id'
        if 'session_id' in df.columns:
            return 'session_id'
        return None
    # Treino
    if 'session_id' in df.columns:
        return 'session_id'
    if 'match_id' in df.columns:
        return 'match_id'
    return None


def _prepare_tracking_times(tracking_df: pd.DataFrame, context_filter: str) -> pd.DataFrame:
    """Normaliza tracking: timestamp -> datetime, cria `period_id` e `t_min` (minutos desde o início)."""
    if tracking_df is None or tracking_df.empty:
        return tracking_df
    if 'player_id' not in tracking_df.columns:
        return tracking_df

    period_col = _tracking_period_col(tracking_df, context_filter)
    if period_col is None:
        return tracking_df

    out = tracking_df.copy()
    if 'timestamp_utc' not in out.columns:
        return out

    out['timestamp_utc'] = pd.to_datetime(out['timestamp_utc'], errors='coerce', utc=True)
    out = out.dropna(subset=['timestamp_utc']).copy()

    out['period_id'] = out[period_col].astype(str)

    # minutos desde o primeiro sample por jogador/período
    first_ts = out.groupby(['period_id', 'player_id'])['timestamp_utc'].transform('min')
    out['t_min'] = (out['timestamp_utc'] - first_ts).dt.total_seconds() / 60.0

    return out


def _distance_from_speed_m(tracking_df: pd.DataFrame) -> pd.Series:
    """Estimativa da distância (m) integrando `inst_speed_mps` ao longo do tempo por jogador/período."""
    if tracking_df is None or tracking_df.empty:
        return pd.Series(dtype=float)
    required = {'period_id', 'player_id', 'timestamp_utc', 'inst_speed_mps'}
    if not required.issubset(set(tracking_df.columns)):
        return pd.Series(dtype=float)

    df = tracking_df[['period_id', 'player_id', 'timestamp_utc', 'inst_speed_mps']].copy()
    df = df.dropna(subset=['timestamp_utc', 'inst_speed_mps'])
    df = df.sort_values(['period_id', 'player_id', 'timestamp_utc'])

    # dt por sample (segundos) dentro de cada (period_id, player_id)
    dt_s = df.groupby(['period_id', 'player_id'])['timestamp_utc'].diff().dt.total_seconds()

    # usar mediana global como fallback para o 1º ponto; limitar para evitar gaps gigantes
    median_dt = float(dt_s.dropna().median()) if dt_s.notna().any() else 5.0
    dt_s = dt_s.fillna(median_dt).clip(lower=0.0, upper=max(10.0, median_dt * 3))

    # distância = v * dt
    dist_m = df['inst_speed_mps'].astype(float).clip(lower=0.0) * dt_s
    return dist_m.groupby([df['period_id'], df['player_id']]).sum()


def _color_to_rgba(color: str, alpha: float) -> str:
    """Converte cores hex/rgb para rgba (para preenchimentos no Plotly)."""
    if not isinstance(color, str):
        return f"rgba(52, 152, 219, {alpha})"
    c = color.strip()
    if c.startswith('rgb('):
        return c.replace('rgb(', 'rgba(').replace(')', f', {alpha})')
    if c.startswith('#') and len(c) == 7:
        try:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
        except:
            return f"rgba(52, 152, 219, {alpha})"
    return f"rgba(52, 152, 219, {alpha})"


def _auto_axis_ranges(
    x_vals,
    y_vals,
    *,
    goal_y: float = 20.0,
    min_half_width: float = 8.0,
    min_y_span: float = 10.0,
):
    """Calcula ranges de eixos (x/y) para auto-centrar o mapa sem cortar baliza/arco."""
    try:
        x_arr = pd.to_numeric(pd.Series(x_vals), errors='coerce').dropna().astype(float)
        y_arr = pd.to_numeric(pd.Series(y_vals), errors='coerce').dropna().astype(float)
    except Exception:
        return [-8, 8], [10, 20.8]

    if x_arr.empty or y_arr.empty:
        return [-8, 8], [10, 20.8]

    # X: simétrico em torno de 0
    x_abs_max = float(np.max(np.abs(x_arr)))
    half_width = max(min_half_width, x_abs_max * 1.15)

    # Y: garantir que inclui a baliza (goal_y) e os pontos/arco
    y_min = float(np.min(y_arr))
    y_max = float(np.max(y_arr))
    y_max = max(y_max, float(goal_y))

    span = max(1e-6, y_max - y_min)
    y_margin = max(0.6, span * 0.12)

    y0 = y_min - y_margin
    y1 = y_max + 0.6

    # garantir uma janela mínima (para evitar zoom excessivo em poucos pontos)
    if (y1 - y0) < min_y_span:
        mid = (y0 + y1) / 2.0
        half = min_y_span / 2.0
        y0, y1 = mid - half, mid + half

    return [-half_width, half_width], [y0, y1]


def _extract_period_number(period_id: str) -> int | None:
    """Extrai o número de IDs tipo match12/session3 para ordenar cronologicamente."""
    if not isinstance(period_id, str):
        return None
    m = re.search(r"(?:match|session)(\d+)", period_id.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _period_sort_key(period_id: str):
    n = _extract_period_number(str(period_id))
    return (n is None, n if n is not None else str(period_id))


def _save_rate_pct(df: pd.DataFrame) -> float:
    """% defesas sobre (defesas + golos)."""
    if df is None or df.empty or 'shot_result' not in df.columns:
        return float('nan')
    valid = df[df['shot_result'].isin(['goal', 'defense'])]
    if valid.empty:
        return float('nan')
    num_def = float((valid['shot_result'] == 'defense').sum())
    num_goal = float((valid['shot_result'] == 'goal').sum())
    denom = num_def + num_goal
    return (num_def / denom * 100.0) if denom > 0 else float('nan')


def _opponent_conversion_pct(df: pd.DataFrame) -> float:
    """% golos sobre (defesas + golos)."""
    if df is None or df.empty or 'shot_result' not in df.columns:
        return float('nan')
    valid = df[df['shot_result'].isin(['goal', 'defense'])]
    if valid.empty:
        return float('nan')
    num_goal = float((valid['shot_result'] == 'goal').sum())
    denom = float(len(valid))
    return (num_goal / denom * 100.0) if denom > 0 else float('nan')


def _goalkeeper_minute_timeseries(gk_df: pd.DataFrame, bin_min: int = 5) -> pd.DataFrame:
    """Série por tempo de jogo (min): defesas/golos + reação média."""
    if gk_df is None or gk_df.empty:
        return pd.DataFrame(columns=['tempo_bin', 'defesas', 'golos', 'save_rate_pct', 'reaction_time_ms'])

    tmp = gk_df[['time', 'shot_result', 'reaction_time']].copy()
    tmp = tmp[tmp['time'].notna() & tmp['shot_result'].notna()].copy()
    tmp['t_min'] = tmp['time'].astype(str).apply(time_to_minutes)
    tmp = tmp.dropna(subset=['t_min'])
    tmp = tmp[tmp['shot_result'].isin(['goal', 'defense'])].copy()
    if tmp.empty:
        return pd.DataFrame(columns=['tempo_bin', 'defesas', 'golos', 'save_rate_pct', 'reaction_time_ms'])

    tmp['tempo_bin'] = (tmp['t_min'].astype(float) // float(bin_min)) * float(bin_min)
    grouped = tmp.groupby('tempo_bin', dropna=True)
    out = grouped.agg(
        defesas=('shot_result', lambda x: (x == 'defense').sum()),
        golos=('shot_result', lambda x: (x == 'goal').sum()),
        reaction_time_ms=('reaction_time', 'mean'),
    ).reset_index()
    denom = (out['defesas'] + out['golos']).astype(float)
    out['save_rate_pct'] = np.where(denom > 0, out['defesas'].astype(float) / denom * 100.0, np.nan)
    return out.sort_values('tempo_bin')


def _team_intensity_timeseries(tracking_df: pd.DataFrame, context_filter: str, bin_min: int = 2) -> pd.DataFrame:
    """Intensidade da equipa ao longo do tempo: média/p75 de inst_speed_mps por minuto."""
    tracking_team = _prepare_tracking_times(tracking_df, context_filter)
    if tracking_team is None or tracking_team.empty or 'inst_speed_mps' not in tracking_team.columns:
        return pd.DataFrame(columns=['tempo_bin', 'mean_speed', 'p75_speed'])
    tmp = tracking_team[['t_min', 'inst_speed_mps']].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=['tempo_bin', 'mean_speed', 'p75_speed'])
    tmp['tempo_bin'] = (tmp['t_min'].astype(float) // float(bin_min)) * float(bin_min)
    out = (
        tmp.groupby('tempo_bin', dropna=True)['inst_speed_mps']
        .agg(mean_speed='mean', p75_speed=lambda s: float(np.nanpercentile(s.astype(float), 75)))
        .reset_index()
        .sort_values('tempo_bin')
    )
    return out


def _period_summary_metrics(gk_df: pd.DataFrame, context_filter: str) -> pd.DataFrame:
    """Resumo por match/session: save%, reação média e conversão adversária."""
    if gk_df is None or gk_df.empty:
        return pd.DataFrame(columns=['period_id', 'save_rate_pct', 'reaction_time_ms', 'opponent_conv_pct', 'shots'])

    period_col = _tracking_period_col(gk_df, context_filter)
    if period_col is None:
        # fallback: usar match_id/session_id se existir
        if 'match_id' in gk_df.columns:
            period_col = 'match_id'
        elif 'session_id' in gk_df.columns:
            period_col = 'session_id'
        else:
            return pd.DataFrame(columns=['period_id', 'save_rate_pct', 'reaction_time_ms', 'opponent_conv_pct', 'shots'])

    tmp = gk_df.copy()
    tmp['period_id'] = tmp[period_col].astype(str)
    tmp_valid = tmp[tmp['shot_result'].isin(['goal', 'defense'])].copy()
    if tmp_valid.empty:
        return pd.DataFrame(columns=['period_id', 'save_rate_pct', 'reaction_time_ms', 'opponent_conv_pct', 'shots'])

    grouped = tmp_valid.groupby('period_id', dropna=True)
    out = grouped.agg(
        shots=('shot_result', 'size'),
        defesas=('shot_result', lambda x: (x == 'defense').sum()),
        golos=('shot_result', lambda x: (x == 'goal').sum()),
        reaction_time_ms=('reaction_time', 'mean'),
    ).reset_index()
    denom = (out['defesas'] + out['golos']).astype(float)
    out['save_rate_pct'] = np.where(denom > 0, out['defesas'].astype(float) / denom * 100.0, np.nan)
    out['opponent_conv_pct'] = np.where(denom > 0, out['golos'].astype(float) / denom * 100.0, np.nan)
    out = out[['period_id', 'save_rate_pct', 'reaction_time_ms', 'opponent_conv_pct', 'shots']]
    out['__sort'] = out['period_id'].map(_period_sort_key)
    out = out.sort_values('__sort').drop(columns='__sort')
    return out

# ============================================================================
# SIDEBAR - PAINEL DE CONTROLO
# ============================================================================
st.sidebar.title("PAINEL DE CONTROLO")
st.sidebar.markdown("---")

# Filtro 1: CONTEXTO
st.sidebar.subheader("Filtro 1: CONTEXTO")
context_filter = st.sidebar.radio(
    "",
    options=["Jogo", "Treino"],
    horizontal=True,
    key="context_filter"
)

# Filtro 2: PERÍODO (Cascata)
st.sidebar.subheader("Filtro 2: PERÍODO")

# Determinar as opções com base no contexto (só matchX no Jogo, só sessionY no Treino)
if context_filter == "Jogo":
    available_periods = _filter_ids_by_prefix(gkdf['match_id'].dropna().unique(), 'match')
else:
    available_periods = _filter_ids_by_prefix(trdf['session_id'].dropna().unique(), 'session')

# Usar IDs reais para o filtro funcionar sem mapeamentos
period_options = ["Todos"] + available_periods

# Se o utilizador tinha um valor antigo em sessão (ex: com Title/underscores), resetar
if "period_filter" in st.session_state and st.session_state["period_filter"] not in period_options:
    st.session_state["period_filter"] = "Todos"

selected_period = st.sidebar.selectbox(
    "Selecione o período",
    options=period_options,
    key="period_filter"
)

# Aplicar filtros de contexto/período a todos os datasets (antes do filtro de guarda-redes)
gk_period = filter_by_context_and_period(gkdf, context_filter, selected_period)
shots_period = filter_by_context_and_period(shotsdf, context_filter, selected_period)
tr_period = filter_by_context_and_period(trdf, context_filter, selected_period)

st.sidebar.markdown("---")

# Filtro de Guarda-Redes
st.sidebar.subheader("🥅 GUARDA-REDES")
goalkeepers = gk_period['goalkeeper'].dropna().unique()
selected_goalkeepers = st.sidebar.multiselect(
    "Selecionar Guarda-Redes",
    options=sorted(goalkeepers),
    default=sorted(goalkeepers),
    key="goalkeeper_filter"
)

# Filtrar dados do(s) guarda-redes selecionado(s) (disponível para todas as abas)
gk_filtered = gk_period[gk_period['goalkeeper'].isin(selected_goalkeepers)].copy()

# Criar abas principais
tab1, tab2, tab3 = st.tabs(["🥅 GUARDA-REDES", "👥 EQUIPA", "📌 CONCLUSÕES"])

# ============================================================================
# ABA 1: GUARDA-REDES
# ============================================================================
with tab1:
    # ========================================================================
    # LINHA 1: KPIs
    # ========================================================================
    st.subheader("Indicadores Principais")

    # KPI 1: Defesas
    num_defesas = len(gk_filtered[gk_filtered['shot_result'] == 'defense'])

    # KPI 2: Golos Sofridos
    num_golos = len(gk_filtered[gk_filtered['shot_result'] == 'goal'])

    # KPI 3: Tempo de Reação Médio
    tempo_reacao_medio = gk_filtered['reaction_time'].mean() if len(gk_filtered) > 0 else np.nan
    tempo_reacao_txt = f"{tempo_reacao_medio:.1f} ms" if pd.notna(tempo_reacao_medio) else "N/A"

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-item">
                <div class="kpi-label">🛡️ Defesas</div>
                <div class="kpi-value">{num_defesas}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">⚠️ Golos Sofridos</div>
                <div class="kpi-value">{num_golos}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">⚡ Tempo de Reação Médio</div>
                <div class="kpi-value">{tempo_reacao_txt}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # ========================================================================
    # LINHA 2: Visualizações Lado a Lado
    # ========================================================================
    # Colocar controlos numa linha própria acima dos gráficos garante alinhamento vertical
    controls_left, controls_right = st.columns(2)

    with controls_left:
        # Filtros específicos (Heatmap)
        filtro_resultado_heat = st.radio(
            "Filtrar:",
            options=['Apenas Golos', 'Golos e Defesas', 'Todos'],
            index=2,
            key="heat_filter",
            horizontal=True,
            label_visibility="collapsed"
        )

    with controls_right:
        # Espaço reservado (mantém a linha de controlos com altura estável)
        st.markdown("&nbsp;", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    # ---- COLUNA ESQUERDA: Heatmap da Baliza ----
    with col_left:
        st.subheader("Heatmap da Baliza: Onde Sofre Golos")

        # Preparar dados
        gk_positions = gk_filtered[
            (gk_filtered['goal_pos_x'].notna()) &
            (gk_filtered['goal_pos_y'].notna())
        ].copy()

        # Aplicar filtro
        if filtro_resultado_heat == 'Apenas Golos':
            gk_positions = gk_positions[gk_positions['shot_result'] == 'goal']
        elif filtro_resultado_heat == 'Golos e Defesas':
            gk_positions = gk_positions[gk_positions['shot_result'].isin(['goal', 'defense'])]

        # Remover 'blocked'
        gk_positions = gk_positions[gk_positions['shot_result'] != 'blocked']

        # Criar figura
        color_map = {
            'goal': 'red',
            'defense': 'blue',
            'post': 'orange',
            'miss': 'green'
        }

        result_names = {
            'goal': 'Golo',
            'defense': 'Defesa',
            'post': 'Poste',
            'miss': 'Fora'
        }

        fig_heat = go.Figure()

        # Desenhar por camadas: primeiro tudo, e por último os "Golos"
        resultados_presentes = [
            r for r in gk_positions['shot_result'].dropna().unique().tolist()
            if r in color_map
        ]
        resultados_sem_golo = [r for r in resultados_presentes if r != 'goal']
        resultados_ordenados = resultados_sem_golo + (['goal'] if 'goal' in resultados_presentes else [])

        for resultado in resultados_ordenados:
            cor = color_map[resultado]
            dados = gk_positions[gk_positions['shot_result'] == resultado]
            if len(dados) > 0:
                fig_heat.add_trace(go.Scatter(
                    x=dados['goal_pos_x'],
                    y=dados['goal_pos_y'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=cor,
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    name=f'{result_names[resultado]} ({len(dados)})'
                ))
        
        if len(gk_positions) > 0:
            x_min, x_max = gk_positions['goal_pos_x'].min(), gk_positions['goal_pos_x'].max()
            y_min, y_max = gk_positions['goal_pos_y'].min(), gk_positions['goal_pos_y'].max()
            x_margin = (x_max - x_min) * 0.08 if x_max != x_min else 0.2
            y_margin = (y_max - y_min) * 0.02 if y_max != y_min else 0.05
        else:
            x_min, x_max, x_margin = -2, 2, 0.2
            y_min, y_max, y_margin = -2, 2, 0.05
        
        # Adicionar imagem de baliza como fundo (se existir)
        # OPÇÃO 1: URL da web
        # fig_heat.add_layout_image(
        #     dict(
        #         source="https://example.com/baliza.png",
        #         xref="x", yref="y",
        #         x=x_min - x_margin, y=y_min - y_margin,
        #         sizex=x_max - x_min + 2*x_margin,
        #         sizey=y_max - y_min + 2*y_margin,
        #         sizing="stretch", opacity=0.3, layer="below"
        #     )
        # )
        
        # OPÇÃO 2: Imagem local (descomente e ajuste o caminho)
        # img_path = Path("baliza.png")
        # if img_path.exists():
        #     img_base64 = get_image_base64(img_path)
        #     if img_base64:
        #         fig_heat.add_layout_image(
        #             dict(
        #                 source=f"data:image/png;base64,{img_base64}",
        #                 xref="x", yref="y",
        #                 x=x_min - x_margin, y=y_min - y_margin,
        #                 sizex=x_max - x_min + 2*x_margin,
        #                 sizey=y_max - y_min + 2*y_margin,
        #                 sizing="stretch", opacity=0.3, layer="below"
        #             )
        #         )
        
        # Adicionar grelha 3x2 representando as zonas da baliza
        # Dimensões da baliza: largura 3m (-1.5 a 1.5), altura 2m (-1 a 1)
        baliza_x_min, baliza_x_max = -1.5, 1.5
        baliza_y_min, baliza_y_max = -1.0, 1.0
        
        # Dividir em 3 colunas e 2 linhas
        col_width = (baliza_x_max - baliza_x_min) / 3  # 2.44m por coluna
        row_height = (baliza_y_max - baliza_y_min) / 2  # 1.22m por linha
        
        # Adicionar linhas verticais (divisão em 3 colunas)
        for i in range(1, 3):  # 2 linhas verticais para criar 3 colunas
            x_pos = baliza_x_min + i * col_width
            fig_heat.add_shape(
                type="line",
                x0=x_pos, y0=baliza_y_min,
                x1=x_pos, y1=baliza_y_max,
                line=dict(color="rgba(100, 100, 100, 0.4)", width=1.5, dash="dash"),
                layer="below"
            )
        
        # Adicionar linha horizontal (divisão em 2 linhas)
        y_pos = baliza_y_min + row_height  # Linha no meio (y=0)
        fig_heat.add_shape(
            type="line",
            x0=baliza_x_min, y0=y_pos,
            x1=baliza_x_max, y1=y_pos,
            line=dict(color="rgba(100, 100, 100, 0.4)", width=1.5, dash="dash"),
            layer="below"
        )
        
        # Adicionar contorno externo da baliza
        fig_heat.add_shape(
            type="rect",
            x0=baliza_x_min, y0=baliza_y_min,
            x1=baliza_x_max, y1=baliza_y_max,
            line=dict(color="rgba(0, 0, 0, 0.6)", width=2),
            fillcolor="rgba(255, 255, 255, 0.05)",
            layer="below"
        )
        
        fig_heat.update_layout(
            xaxis_title='Posição X na Baliza',
            yaxis_title='Posição Y na Baliza',
            xaxis=dict(range=[x_min - x_margin, x_max + x_margin]),
            yaxis=dict(range=[y_min - y_margin, y_max + y_margin], scaleanchor="x", scaleratio=1),
            height=280,
            margin=dict(t=20, b=65, l=50, r=140),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02
            )
        )
        
        st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CONFIG)
    
    # ---- COLUNA DIREITA: Scatter Reação vs Cansaço ----
    with col_right:
        st.subheader("Tempo de Reação vs Cansaço")
        
        # Preparar dados
        time_reaction_data = gk_filtered[['goalkeeper', 'time', 'reaction_time']].copy()
        time_reaction_data = time_reaction_data[
            (time_reaction_data['reaction_time'].notna()) &
            (time_reaction_data['time'].notna())
        ].copy()
        
        time_reaction_data['tempo_jogo_min'] = time_reaction_data['time'].apply(time_to_minutes)
        time_reaction_data = time_reaction_data.dropna(subset=['tempo_jogo_min'])
        
        if len(time_reaction_data) > 0:
            fig_scatter = go.Figure()
            
            color_palette = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
            unique_gk = sorted(time_reaction_data['goalkeeper'].unique())
            gk_colors = {gk: color_palette[i % len(color_palette)] for i, gk in enumerate(unique_gk)}
            
            for gk in unique_gk:
                subset = time_reaction_data[time_reaction_data['goalkeeper'] == gk]

                subset_sorted = subset.sort_values('tempo_jogo_min')

                # Apenas linhas: média móvel (janela 5) quando possível; caso contrário linha simples
                if len(subset_sorted) >= 5:
                    rolling_mean = subset_sorted['reaction_time'].rolling(window=5, center=True).mean()
                    y_series = rolling_mean
                    trace_name = f'{gk} (média móvel)'
                else:
                    y_series = subset_sorted['reaction_time']
                    trace_name = f'{gk}'

                fig_scatter.add_trace(go.Scatter(
                    x=subset_sorted['tempo_jogo_min'],
                    y=y_series,
                    mode='lines',
                    name=trace_name,
                    line=dict(
                        color=gk_colors[gk],
                        width=3
                    ),
                    text=[f"{gk}<br>{t:.1f} min<br>{r:.0f} ms"
                          for t, r in zip(subset_sorted['tempo_jogo_min'], subset_sorted['reaction_time'])],
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True
                ))
            
            fig_scatter.update_layout(
                xaxis_title='Tempo de Jogo (min)',
                yaxis_title='Tempo de Reação (ms)',
                height=280,
                margin=dict(t=20, b=65, l=50, r=170),
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.02
                )
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Sem dados suficientes para exibir.")
    
    # ========================================================================
    # LINHA 3: Evolução Acumulada dos Resultados ao Longo do Jogo
    # ========================================================================
    st.subheader("Evolução Acumulada dos Resultados de Remate ao Longo do Jogo")
    
    # Preparar dados
    shot_result_timeline = gk_filtered[['time', 'shot_result']].copy()
    shot_result_timeline = shot_result_timeline[
        (shot_result_timeline['time'].notna()) &
        (shot_result_timeline['shot_result'].notna())
    ].copy()
    
    shot_result_timeline['tempo_jogo_min'] = shot_result_timeline['time'].apply(time_to_minutes)
    shot_result_timeline = shot_result_timeline.dropna(subset=['tempo_jogo_min'])
    shot_result_timeline = shot_result_timeline.sort_values('tempo_jogo_min')
    
    if len(shot_result_timeline) > 0:
        # Criar intervalos de tempo (bins de 2 minutos)
        shot_result_timeline['tempo_bin'] = (shot_result_timeline['tempo_jogo_min'] // 2) * 2
        
        # Contar ocorrências por intervalo e resultado
        result_counts = shot_result_timeline.groupby(['tempo_bin', 'shot_result']).size().reset_index(name='count')
        result_pivot = result_counts.pivot(index='tempo_bin', columns='shot_result', values='count').fillna(0)
        result_cumsum = result_pivot.cumsum()
        
        # Cores e nomes em português
        color_map_results = {
            'goal': '#E74C3C',
            'defense': '#3498DB',
            'post': '#F39C12',
            'blocked': '#9B59B6',
            'miss': '#95A5A6'
        }
        
        result_names_pt = {
            'goal': 'Golo',
            'defense': 'Defesa',
            'post': 'Poste',
            'blocked': 'Bloqueado',
            'miss': 'Fora'
        }
        
        fig_evolution = go.Figure()
        
        for result in result_cumsum.columns:
            fig_evolution.add_trace(go.Scatter(
                x=result_cumsum.index,
                y=result_cumsum[result],
                mode='lines',
                name=result_names_pt.get(result, result),
                stackgroup='one',
                fillcolor=color_map_results.get(result, '#999999'),
                line=dict(width=0.5)
            ))
        
        fig_evolution.update_layout(
            margin=dict(t=5, b=40, l=105, r=20),
            xaxis_title='Tempo de Jogo (minutos)',
            yaxis=dict(title=dict(text='Contagem Acumulada', standoff=12), automargin=True),
            xaxis=dict(automargin=True),
            height=190,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_evolution, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.info("Sem dados suficientes para exibir.")

# ============================================================================
# ABA 2: EQUIPA
# ============================================================================
with tab2:
    # Fazer merge dos dados
    team_data = shots_period.merge(gk_filtered[['shot_id', 'shot_result']], on='shot_id', how='left')
    team_data = team_data[team_data['shot_result'].notna()]
    
    # ========================================================================
    # LINHA 1: KPIs
    # ========================================================================
    st.subheader("Indicadores Principais")

    # KPI 1: Remates Sofridos
    num_remates = len(team_data)

    # KPI 3: Eficácia Adversária (% golos)
    num_golos_equipa = len(team_data[team_data['shot_result'] == 'goal'])
    eficacia = (num_golos_equipa / num_remates * 100) if num_remates > 0 else 0

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-item">
                <div class="kpi-label">🎯 Remates Sofridos</div>
                <div class="kpi-value">{num_remates}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">⚡ Eficácia Adversária</div>
                <div class="kpi-value">{eficacia:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # ========================================================================
    # LINHA 2: Gráficos (linha de cima)
    # ========================================================================
    row_top_left, row_top_right = st.columns(2)

    # ---- COLUNA ESQUERDA (TOPO): Mapa de Campo ----
    with row_top_left:
        st.subheader("Mapa de Campo: Zonas de Remate")
        
        # Filtro de resultado
        filtro_mapa = st.radio(
            "Filtrar:",
            options=['Apenas Golos', 'Golos e Defesas', 'Todos'],
            index=2,
            key="mapa_filter",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Preparar dados
        mapa_data = team_data[
            (team_data['shot_pos_x'].notna()) &
            (team_data['shot_pos_y'].notna())
        ].copy()

        # Rodar o mapa: baliza fica no topo (centrada) e a semi-circunferência à volta
        # Novo sistema: x = lateral (antigo Y), y = distância à baliza (=-antigo X)
        mapa_data['plot_x'] = mapa_data['shot_pos_y'].astype(float)
        mapa_data['plot_y'] = (-mapa_data['shot_pos_x']).astype(float)
        
        # Aplicar filtro
        if filtro_mapa == 'Apenas Golos':
            mapa_data = mapa_data[mapa_data['shot_result'] == 'goal']
        elif filtro_mapa == 'Golos e Defesas':
            mapa_data = mapa_data[mapa_data['shot_result'].isin(['goal', 'defense'])]
        
        # Remover 'blocked'
        mapa_data = mapa_data[mapa_data['shot_result'] != 'blocked']
        
        # Definir categorias e cores
        result_names_all = {
            'goal': 'Golo', 'defense': 'Defesa', 'post': 'Poste',
            'blocked': 'Bloqueado', 'miss': 'Fora'
        }
        mapa_data['categoria'] = mapa_data['shot_result'].map(result_names_all)
        color_map_mapa = {
            'Golo': 'red', 'Defesa': 'blue', 'Poste': 'orange',
            'Bloqueado': 'purple', 'Fora': 'green'
        }
        
        fig_mapa = go.Figure()

        # Desenhar os pontos por camadas: primeiro tudo, e por último os "Golos"
        categorias = [c for c in mapa_data['categoria'].dropna().unique().tolist()]
        categorias_sem_golo = [c for c in categorias if c != 'Golo']
        categorias_ordenadas = categorias_sem_golo + (['Golo'] if 'Golo' in categorias else [])

        for categoria in categorias_ordenadas:
            dados_cat = mapa_data[mapa_data['categoria'] == categoria]
            fig_mapa.add_trace(go.Scatter(
                x=dados_cat['plot_x'],
                y=dados_cat['plot_y'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color_map_mapa.get(categoria, 'gray'),
                    line=dict(width=1, color='white'),
                    opacity=0.7
                ),
                name=f'{categoria} ({len(dados_cat)})'
            ))
        
        # Desenhar área dos 6 metros
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        x_circle = -20 + 6 * np.cos(theta)
        y_circle = 0 + 6 * np.sin(theta)

        # Aplicar a mesma rotação às formas
        x_circle_plot = y_circle
        y_circle_plot = -x_circle

        fig_mapa.add_trace(go.Scatter(
            x=x_circle_plot, y=y_circle_plot,
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Área 6m', showlegend=True
        ))
        
        # Linha da baliza
        fig_mapa.add_trace(go.Scatter(
            x=[-1.5, 1.5], y=[20, 20],
            mode='lines',
            line=dict(color='black', width=3),
            name='Baliza', showlegend=True
        ))

        # Auto-centro dinâmico: inclui pontos + arco 6m + baliza
        goal_y = 20.0
        x_all = pd.concat(
            [
                mapa_data['plot_x'].astype(float),
                pd.Series(x_circle_plot, dtype=float),
                pd.Series([-1.5, 1.5], dtype=float),
            ],
            ignore_index=True,
        )
        y_all = pd.concat(
            [
                mapa_data['plot_y'].astype(float),
                pd.Series(y_circle_plot, dtype=float),
                pd.Series([goal_y, goal_y], dtype=float),
            ],
            ignore_index=True,
        )

        x_range, y_range = _auto_axis_ranges(x_all, y_all, goal_y=goal_y)

        # Zoom horizontal: mantém o centro e reduz o range do X (mais visibilidade dos pontos)
        if x_range and len(x_range) == 2 and MAPA_X_ZOOM and 0 < MAPA_X_ZOOM < 1:
            x0 = float(x_range[0])
            x1 = float(x_range[1])
            x_center = (x0 + x1) / 2.0
            half = max(1e-6, (x1 - x0) / 2.0)
            new_half = half * float(MAPA_X_ZOOM)
            x_range = [x_center - new_half, x_center + new_half]

        # Zoom vertical: mantém X igual e reduz o range do Y (mais visibilidade dos pontos)
        # Mantém o limite superior (perto da baliza) e aproxima o limite inferior.
        if y_range and len(y_range) == 2 and MAPA_Y_ZOOM and 0 < MAPA_Y_ZOOM < 1:
            y_max = float(y_range[1])
            y_min = float(y_range[0])
            span = max(1e-6, y_max - y_min)
            new_span = span * float(MAPA_Y_ZOOM)
            y_range = [y_max - new_span, y_max]
        
        fig_mapa.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            xaxis=dict(range=x_range, showgrid=False, zeroline=False, showticklabels=False),
            # baliza em y=20; range dinâmico mantém tudo visível e centra melhor
            yaxis=dict(range=y_range, showgrid=False, zeroline=False, showticklabels=False),
            height=200,
            margin=dict(t=10, b=55, l=45, r=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="left",
                x=0.0,
                font=dict(size=10),
            ),
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        
        st.plotly_chart(fig_mapa, use_container_width=True, config=PLOTLY_CONFIG)

    # ---- COLUNA DIREITA (TOPO): Top Jogadores Perigosos ----
    with row_top_right:
        st.subheader("Top Jogadores Adversários Perigosos")
        
        # Número de jogadores a exibir
        num_top = st.slider("Número de jogadores:", 3, 20, 3, key="top_players")
        
        # Calcular estatísticas por jogador
        player_stats = team_data.groupby('player_id').agg({
            'shot_id': 'count',
            'shot_result': lambda x: (x == 'goal').sum()
        }).reset_index()
        player_stats.columns = ['player_id', 'total_shots', 'goals']
        player_stats['conversion_rate'] = (
            player_stats['goals'] / player_stats['total_shots'] * 100
        ).round(1)
        
        player_stats = player_stats.sort_values('goals', ascending=False).head(num_top)
        
        if len(player_stats) > 0:
            fig_players = go.Figure()

            # Cores por ranking: 1º dourado, 2º prateado, 3º bronze, resto cor padrão
            podium_colors = {
                0: '#D4AF37',  # dourado
                1: '#C0C0C0',  # prateado
                2: '#CD7F32',  # bronze
            }
            default_color = '#3498DB'
            bar_colors = [podium_colors.get(i, default_color) for i in range(len(player_stats))]

            fig_players.add_trace(go.Bar(
                x=player_stats['player_id'].astype(str),
                y=player_stats['goals'],
                marker=dict(color=bar_colors),
                text=[f"{int(g)} golos ({cr}%)"
                      for g, cr in zip(player_stats['goals'], player_stats['conversion_rate'])],
                textposition='inside',
                textfont=dict(color='white', size=11),
                cliponaxis=True,
                hovertemplate='<b>Jogador %{x}</b><br>Golos: %{y}<br><extra></extra>'
            ))
            
            fig_players.update_layout(
                xaxis_title='Jogador (ID)',
                yaxis_title=None,
                height=195,
                showlegend=False,
                bargap=0,
                bargroupgap=0,
                xaxis=dict(tickangle=-20, showgrid=False),
                margin=dict(t=20, b=35, l=50, r=20),
                yaxis=dict(
                    range=[0, float(player_stats['goals'].max()) + max(1.0, float(player_stats['goals'].max()) * 0.2)],
                    showgrid=False,
                    zeroline=False
                ),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_players, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            pass

    # ========================================================================
    # LINHA 3: Gráficos (linha de baixo) — garante alinhamento entre colunas
    # ========================================================================
    row_bottom_left, row_bottom_right = st.columns(2)

    # ---- COLUNA ESQUERDA (BAIXO): Ridgeline ----
    with row_bottom_left:
        st.subheader("Ridgeline: Velocidade Instantânea ao Longo do Tempo")

        tracking_team = _prepare_tracking_times(tr_period, context_filter)

        if tracking_team is None or tracking_team.empty or 'inst_speed_mps' not in tracking_team.columns:
            st.info("Sem dados de tracking suficientes para exibir.")
        else:
            unique_players = sorted(tracking_team['player_id'].dropna().astype(str).unique())
            max_players = max(3, min(20, len(unique_players)))

            num_ridge = st.slider(
                "Número de jogadores:",
                3, max_players, 3,
                key="ridge_players"
            )

            # Top jogadores por distância (estimada) para tornar o ridgeline legível
            dist_series = _distance_from_speed_m(tracking_team)
            dist_df = dist_series.reset_index(name='dist_m') if not dist_series.empty else pd.DataFrame(columns=['period_id', 'player_id', 'dist_m'])
            if selected_period == 'Todos':
                dist_rank = dist_df.groupby('player_id', dropna=True)['dist_m'].mean()
            else:
                dist_rank = dist_df.groupby('player_id', dropna=True)['dist_m'].sum()

            top_players_ridge = (
                dist_rank.sort_values(ascending=False)
                .head(num_ridge)
                .index.astype(str)
                .tolist()
            )

            ridge_data = tracking_team[tracking_team['player_id'].astype(str).isin(top_players_ridge)].copy()
            ridge_data = ridge_data.dropna(subset=['t_min', 'inst_speed_mps'])

            if ridge_data.empty:
                st.info("Sem dados suficientes para o ridgeline.")
            else:
                # Definições fixas (sem UI):
                # - suavização = 0 (sem binning)
                # - amplificação vertical = 3 (máxima)
                ridge_amp_val = 3.0
                ridge_center = True
                bin_size = 0.0

                # bin (em minutos) para suavizar (mantém eixo X em minutos reais)
                if bin_size and bin_size > 0:
                    ridge_data['t_bin_min'] = (
                        np.floor(ridge_data['t_min'].astype(float) / bin_size) * bin_size
                    ).round(2)
                else:
                    # sem binning: usar o tempo em minutos diretamente
                    ridge_data['t_bin_min'] = ridge_data['t_min'].astype(float).round(2)
                speed_bins = (
                    ridge_data.groupby(['period_id', 'player_id', 't_bin_min'])['inst_speed_mps']
                    .mean()
                    .reset_index()
                )

                if selected_period == 'Todos':
                    speed_plot = (
                        speed_bins.groupby(['player_id', 't_bin_min'])['inst_speed_mps']
                        .mean()
                        .reset_index()
                    )
                    plot_title_suffix = "(média)"
                else:
                    speed_plot = (
                        speed_bins.groupby(['player_id', 't_bin_min'])['inst_speed_mps']
                        .mean()
                        .reset_index()
                    )
                    plot_title_suffix = ""

                # Série ajustada para realçar diferenças (apenas para visualização)
                speed_plot['inst_speed_mps'] = speed_plot['inst_speed_mps'].astype(float).clip(lower=0.0)
                if ridge_center:
                    player_mean = speed_plot.groupby('player_id', dropna=False)['inst_speed_mps'].transform('mean')
                    speed_plot['inst_speed_display_mps'] = (player_mean + (speed_plot['inst_speed_mps'] - player_mean) * ridge_amp_val).clip(lower=0.0)
                else:
                    speed_plot['inst_speed_display_mps'] = (speed_plot['inst_speed_mps'] * ridge_amp_val).clip(lower=0.0)

                max_speed = float(speed_plot['inst_speed_display_mps'].max()) if len(speed_plot) else 0.0
                offset = max(1.0, max_speed * 1.4)

                fig_ridge = go.Figure()
                y_tickvals = []
                y_ticktext = []

                # manter ordem por ranking
                ordered_players = [p for p in top_players_ridge if p in set(speed_plot['player_id'].astype(str))]
                palette = px.colors.qualitative.Set2

                for i, player in enumerate(ordered_players):
                    sub = speed_plot[speed_plot['player_id'].astype(str) == str(player)].sort_values('t_bin_min')
                    if sub.empty:
                        continue

                    x_vals = sub['t_bin_min']
                    base = np.full(len(sub), i * offset)
                    y_vals = base + sub['inst_speed_display_mps'].astype(float).clip(lower=0.0)

                    y_tickvals.append(i * offset)
                    y_ticktext.append(str(player))

                    color = palette[i % len(palette)]

                    # trace base (invisível) para permitir fill
                    fig_ridge.add_trace(go.Scatter(
                        x=x_vals,
                        y=base,
                        mode='lines',
                        line=dict(width=0),
                        hoverinfo='skip',
                        showlegend=False
                    ))

                    fig_ridge.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines',
                        line=dict(color=color, width=2),
                        fill='tonexty',
                        fillcolor=_color_to_rgba(color, 0.25),
                        hovertemplate=(
                            f"<b>{player}</b><br>Minuto: %{{x:.2f}}"
                            f"<br>Velocidade: %{{customdata[0]:.2f}} m/s"
                            f"<br>Vel. (realçada): %{{customdata[1]:.2f}} m/s<extra></extra>"
                        ),
                        customdata=np.stack(
                            [
                                sub['inst_speed_mps'].to_numpy(dtype=float),
                                sub['inst_speed_display_mps'].to_numpy(dtype=float),
                            ],
                            axis=-1,
                        ),
                        showlegend=False
                    ))

                fig_ridge.update_layout(
                    height=210,
                    margin=dict(t=10, b=30, l=60, r=20),
                    xaxis_title='Tempo (min)',
                    yaxis_title='Jogador',
                    hovermode='x unified'
                )
                fig_ridge.update_yaxes(
                    tickmode='array',
                    tickvals=y_tickvals,
                    ticktext=y_ticktext,
                    showgrid=False,
                    zeroline=False
                )
                fig_ridge.update_xaxes(dtick=1)

                st.plotly_chart(fig_ridge, use_container_width=True, config=PLOTLY_CONFIG)

    # ---- COLUNA DIREITA (BAIXO): Distância ----
    with row_bottom_right:
        st.subheader("Distância Percorrida")

        tracking_team = _prepare_tracking_times(tr_period, context_filter)

        if tracking_team is None or tracking_team.empty or 'inst_speed_mps' not in tracking_team.columns:
            st.info("Sem dados de tracking suficientes para exibir.")
        else:
            unique_players = sorted(tracking_team['player_id'].dropna().astype(str).unique())
            max_players = max(3, min(20, len(unique_players)))

            num_dist = st.slider(
                "Número de jogadores:",
                3, max_players, 3,
                key="dist_players"
            )

            dist_series = _distance_from_speed_m(tracking_team)
            if dist_series.empty:
                st.info("Sem dados suficientes para calcular distância.")
            else:
                dist_df = dist_series.reset_index(name='dist_m')

                if selected_period == 'Todos':
                    dist_summary = dist_df.groupby('player_id', dropna=True)['dist_m'].mean()
                    title_suffix = "(média por período)"
                else:
                    dist_summary = dist_df.groupby('player_id', dropna=True)['dist_m'].sum()
                    title_suffix = ""

                top_dist = (
                    dist_summary.sort_values(ascending=False)
                    .head(num_dist)
                    .reset_index()
                )
                top_dist.columns = ['player_id', 'dist_m']
                top_dist['dist_km'] = (top_dist['dist_m'].astype(float) / 1000.0).round(2)

                # Para barras horizontais, mostrar o maior em cima
                top_dist = top_dist.sort_values('dist_km', ascending=True)

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Bar(
                    x=top_dist['dist_km'],
                    y=top_dist['player_id'].astype(str),
                    orientation='h',
                    marker=dict(color='#2ECC71'),
                    text=top_dist['dist_km'].map(lambda v: f"{v:.2f} km"),
                    textposition='inside',
                    textfont=dict(color='white', size=11),
                    cliponaxis=True,
                    hovertemplate='<b>%{y}</b><br>Distância: %{x:.2f} km<extra></extra>'
                ))

                fig_dist.update_layout(
                    # usar o subheader como título; reduz espaço interno e ajuda alinhamento
                    title_text="",
                    xaxis_title='Distância (km)',
                    yaxis_title='Jogador (ID)',
                    height=235,
                    margin=dict(t=10, b=35, l=50, r=20),
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                )
                fig_dist.update_xaxes(
                    range=[0, float(top_dist['dist_km'].max()) + max(0.2, float(top_dist['dist_km'].max()) * 0.2)]
                )

                st.plotly_chart(fig_dist, use_container_width=True, config=PLOTLY_CONFIG)

# ============================================================================
# ABA 3: CONCLUSÕES (STORYTELLING)
# ============================================================================
with tab3:
    st.subheader("Resumo & Storytelling")

    # Dados de base (equipa deve considerar todos os guarda-redes do período)
    gk_all_period = gk_period.copy()
    team_data_all = shots_period.merge(
        gk_all_period[['shot_id', 'shot_result']],
        on='shot_id',
        how='left'
    )
    team_data_all = team_data_all[team_data_all['shot_result'].notna()].copy()

    # KPIs principais (simples)
    gk_save = _save_rate_pct(gk_filtered)
    gk_react = float(gk_filtered['reaction_time'].mean()) if (gk_filtered is not None and len(gk_filtered)) else float('nan')
    opp_conv = _opponent_conversion_pct(gk_all_period)

    # Intensidade (tracking)
    intensity_ts = _team_intensity_timeseries(tr_period, context_filter, bin_min=2)
    team_speed = float(intensity_ts['mean_speed'].mean()) if (intensity_ts is not None and len(intensity_ts)) else float('nan')

    gk_save_txt = f"{gk_save:.1f}%" if pd.notna(gk_save) else "N/A"
    gk_react_txt = f"{gk_react:.1f} ms" if pd.notna(gk_react) else "N/A"
    opp_conv_txt = f"{opp_conv:.1f}%" if pd.notna(opp_conv) else "N/A"
    team_speed_txt = f"{team_speed:.2f} m/s" if pd.notna(team_speed) else "N/A"

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-item">
                <div class="kpi-label">🥅 Taxa de Defesa (GR selecionado)</div>
                <div class="kpi-value">{gk_save_txt}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">⚡ Reação Média (GR selecionado)</div>
                <div class="kpi-value">{gk_react_txt}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">🎯 Eficácia Adversária (equipa)</div>
                <div class="kpi-value">{opp_conv_txt}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">🏃 Intensidade Média (equipa)</div>
                <div class="kpi-value">{team_speed_txt}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Evolução por período (match/session) — logo no topo
    per_gk_sel = _period_summary_metrics(gk_filtered, context_filter)
    per_team = _period_summary_metrics(gk_all_period, context_filter)

    if len(per_team) >= 2:
        st.subheader("Evolução por período (match/session)")
        fig_per = go.Figure()
        fig_per.add_trace(go.Scatter(
            x=per_team['period_id'],
            y=per_team['opponent_conv_pct'],
            mode='lines+markers',
            name='Eficácia adversária (%)',
            line=dict(color='#E74C3C', width=3),
            hovertemplate='%{x}<br>Eficácia: %{y:.1f}%<extra></extra>'
        ))
        if len(per_gk_sel) >= 2:
            fig_per.add_trace(go.Scatter(
                x=per_gk_sel['period_id'],
                y=per_gk_sel['save_rate_pct'],
                mode='lines+markers',
                name='Taxa de defesa (GR) (%)',
                line=dict(color='#2ECC71', width=3),
                hovertemplate='%{x}<br>Taxa defesa: %{y:.1f}%<extra></extra>'
            ))

        # Zoom vertical automático para tornar a interseção mais visível
        y_series = [per_team['opponent_conv_pct']]
        if len(per_gk_sel) >= 2:
            y_series.append(per_gk_sel['save_rate_pct'])
        y_all = pd.to_numeric(pd.concat(y_series, ignore_index=True), errors='coerce').dropna().astype(float)
        if len(y_all) > 0:
            y_min = float(y_all.min())
            y_max = float(y_all.max())
            span = max(1e-6, y_max - y_min)
            pad = max(3.0, span * 0.15)
            y0 = max(0.0, y_min - pad)
            y1 = min(100.0, y_max + pad)
            # garantir janela mínima para não ficar demasiado "apertado"
            if (y1 - y0) < 18.0:
                mid = (y0 + y1) / 2.0
                y0 = max(0.0, mid - 9.0)
                y1 = min(100.0, mid + 9.0)
            y_range = [y0, y1]
        else:
            y_range = [0, 100]

        fig_per.update_layout(
            height=220,
            margin=dict(t=10, b=60, l=45, r=20),
            xaxis_title=None,
            yaxis_title='%',
            xaxis=dict(showgrid=False, tickangle=-20, tickfont=dict(size=10)),
            yaxis=dict(range=y_range, showgrid=False),
            legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='left', x=0.0),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_per, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")

    colA, colB = st.columns(2)

    # ---------- ESQ: Evolução do Guarda-Redes ----------
    with colA:
        st.subheader("Guarda-redes: evolução ao longo do tempo")

        gk_ts = _goalkeeper_minute_timeseries(gk_filtered, bin_min=5)
        if gk_ts.empty:
            st.info("Sem dados suficientes para construir a evolução do guarda-redes.")
        else:
            fig_gk = go.Figure()
            fig_gk.add_trace(go.Scatter(
                x=gk_ts['tempo_bin'],
                y=gk_ts['save_rate_pct'],
                mode='lines+markers',
                name='Taxa de defesa (%)',
                line=dict(color='#2ECC71', width=3),
                hovertemplate='Minuto %{x:.0f}–%{x:.0f}+5<br>Taxa defesa: %{y:.1f}%<extra></extra>'
            ))
            fig_gk.update_layout(
                height=170,
                margin=dict(t=10, b=35, l=45, r=20),
                xaxis_title='Tempo (min)',
                yaxis_title='Taxa de defesa (%)',
                yaxis=dict(range=[0, 100], showgrid=False),
                xaxis=dict(showgrid=False, dtick=10, tickfont=dict(size=10)),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_gk, use_container_width=True, config=PLOTLY_CONFIG)

    # ---------- DIR: Equipa (eficácia adversária + intensidade) ----------
    with colB:
        st.subheader("Equipa: controlo defensivo & intensidade")

        # Eficácia adversária ao longo do tempo via GK (mais consistente com o relógio do jogo)
        team_gk_ts = _goalkeeper_minute_timeseries(gk_all_period, bin_min=5)
        if team_gk_ts.empty:
            st.info("Sem dados suficientes para construir a evolução da equipa.")
        else:
            # conversão adversária = 100 - save_rate
            conv = 100.0 - team_gk_ts['save_rate_pct']
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=team_gk_ts['tempo_bin'],
                y=conv,
                mode='lines+markers',
                name='Eficácia adversária (%)',
                line=dict(color='#E74C3C', width=3),
                hovertemplate='Minuto %{x:.0f}–%{x:.0f}+5<br>Eficácia adversária: %{y:.1f}%<extra></extra>'
            ))
            fig_conv.update_layout(
                height=170,
                margin=dict(t=10, b=35, l=45, r=20),
                xaxis_title='Tempo (min)',
                yaxis_title='Eficácia adversária (%)',
                yaxis=dict(range=[0, 100], showgrid=False),
                xaxis=dict(showgrid=False, dtick=10, tickfont=dict(size=10)),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_conv, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    st.subheader("Observações (automáticas)")

    observations: list[str] = []

    def _fmt_pp(delta: float) -> str:
        if pd.isna(delta):
            return "N/A"
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f} pp"

    def _fmt_ms(delta: float) -> str:
        if pd.isna(delta):
            return "N/A"
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f} ms"

    # 1) Dentro do período: início vs fim (minutos)
    if (selected_period is not None) and (selected_period != 'Todos'):
        gk_ts = _goalkeeper_minute_timeseries(gk_filtered, bin_min=5)
        if len(gk_ts) >= 4:
            n = len(gk_ts)
            early = gk_ts.head(max(1, n // 3))
            late = gk_ts.tail(max(1, n // 3))
            early_save = float(early['save_rate_pct'].mean())
            late_save = float(late['save_rate_pct'].mean())
            observations.append(
                f"Guarda-redes: taxa de defesa média aumentou de {early_save:.1f}% para {late_save:.1f}% ({_fmt_pp(late_save - early_save)})."
            )

        team_ts = _goalkeeper_minute_timeseries(gk_all_period, bin_min=5)
        if len(team_ts) >= 4 and team_ts['save_rate_pct'].notna().any():
            n = len(team_ts)
            early = team_ts.head(max(1, n // 3))
            late = team_ts.tail(max(1, n // 3))
            early_conv = float((100.0 - early['save_rate_pct']).mean())
            late_conv = float((100.0 - late['save_rate_pct']).mean())
            observations.append(
                f"Equipa: eficácia adversária desceu de {early_conv:.1f}% para {late_conv:.1f}% ({_fmt_pp(late_conv - early_conv)}; menor é melhor)."
            )

    # 2) Entre períodos: comparar início vs fim (match/session)
    else:
        if len(per_gk_sel) >= 2:
            first = per_gk_sel.iloc[0]
            last = per_gk_sel.iloc[-1]
            if pd.notna(first['save_rate_pct']) and pd.notna(last['save_rate_pct']):
                observations.append(
                    f"Guarda-redes (entre períodos): taxa de defesa {first['save_rate_pct']:.1f}% → {last['save_rate_pct']:.1f}% ({_fmt_pp(float(last['save_rate_pct'] - first['save_rate_pct']))})."
                )

        if len(per_team) >= 2:
            first = per_team.iloc[0]
            last = per_team.iloc[-1]
            if pd.notna(first['opponent_conv_pct']) and pd.notna(last['opponent_conv_pct']):
                observations.append(
                    f"Equipa (entre períodos): eficácia adversária {first['opponent_conv_pct']:.1f}% → {last['opponent_conv_pct']:.1f}% ({_fmt_pp(float(last['opponent_conv_pct'] - first['opponent_conv_pct']))})."
                )

    if observations:
        st.markdown("\n".join([f"- {o}" for o in observations]))
    else:
        st.info("Sem dados suficientes para gerar observações.")

# Footer
st.markdown("---")
st.markdown("**Dashboard desenvolvido para análise de performance em andebol**")
