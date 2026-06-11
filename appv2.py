import streamlit as st
import pandas as pd
import numpy as np
import pvlib
from pvlib import shading
import plotly.graph_objects as go
from timezonefinder import TimezoneFinder
import cdsapi
import xarray as xr
import os
import zipfile

# Configuración de página de Streamlit
st.set_page_config(page_title="Motor Fotovoltaico & Demanda Industrial", layout="wide")

st.markdown("""
<style>
[data-testid="stHeaderActionElements"] { display: none; }
</style>
<div style="margin-bottom:1.5rem;">
    <h2 style="margin-bottom:0.4rem;font-weight:700;">
        Motor de Jensen: Simulación Fotovoltaica vs. Demanda Industrial
    </h2>
    <p style="color:#64748b;font-size:0.9rem;margin:0;">
        Herramienta interactiva para dimensionamiento de sistemas FV y análisis de cobertura de carga quinceminutal anual.
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES — DATOS HISTÓRICOS REALES STREGER S.A.
# =============================================================================

# kWh mensuales reales extraídos del recibo CFE (mayo 2025 – mayo 2026)
# MAY usa el promedio de MAY-25 (11,040) y MAY-26 (9,520)
STREGER_KWH_MENSUAL = {
    1:  5_280,   # ENE 26
    2:  8_400,   # FEB 26
    3:  7_600,   # MAR 26
    4:  8_800,   # ABR 26
    5: 10_280,   # MAY — promedio MAY-25 / MAY-26
    6: 10_240,   # JUN 25
    7:  8_080,   # JUL 25
    8:  8_960,   # AGO 25
    9: 10_560,   # SEP 25
    10: 10_240,  # OCT 25
    11:  6_400,  # NOV 25
    12:  7_840,  # DIC 25
}

STREGER_DEMANDA_MAX_KW = 80   # 1 unidad medida × multiplicador 80
STREGER_PRECIO_MEDIO   = 2.67 # $/kWh histórico (sin IVA), del análisis del recibo

# =============================================================================
# FUNCIONES DE SIMULACIÓN (BACKEND)
# =============================================================================

def calcular_tilt_optimo(lat):
    """Ángulo óptimo de inclinación ≈ latitud absoluta del lugar."""
    return int(round(abs(lat)))


@st.cache_data
def calcular_seccion1(lat, lon, alt, tz, tilt, azimut):
    """
    Calcula irradiancia GHI/DNI/DHI y POA para un día típico (21 Jun),
    más estimación rápida de producción de un panel estándar.
    """
    location    = pvlib.location.Location(lat, lon, tz=tz, altitude=alt)
    tiempos_dia = pd.date_range(
        start='2026-06-21 00:00',
        end='2026-06-21 23:45',
        freq='15min',
        tz=tz
    )
    sol_dia      = location.get_solarposition(tiempos_dia)
    clearsky_dia = location.get_clearsky(tiempos_dia, model='ineichen')

    poa_dia = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimut,
        solar_zenith=sol_dia['apparent_zenith'],
        solar_azimuth=sol_dia['azimuth'],
        dni=clearsky_dia['dni'],
        ghi=clearsky_dia['ghi'],
        dhi=clearsky_dia['dhi']
    )['poa_global']

    AREA_DEFAULT = 1.95
    EF_DEFAULT   = 0.218
    kwh_1panel   = float((poa_dia * AREA_DEFAULT * EF_DEFAULT / 1000 * (15 / 60)).sum())

    return tiempos_dia, clearsky_dia, poa_dia, kwh_1panel


def generar_demanda_industrial(demanda_max, factor_planta, tiempos):
    """
    Genera una curva de carga sintética anual cada 15 minutos (35,040 periodos)
    diferenciando Verano, No Verano y Fines de Semana.
    """
    mes        = tiempos.month
    dia_semana = tiempos.weekday
    hora       = tiempos.hour + tiempos.minute / 60.0

    perfil_base = np.zeros(len(tiempos))

    for i, (h, m, ds) in enumerate(zip(hora, mes, dia_semana)):
        es_verano     = 5 <= m <= 9
        es_fin_semana = ds >= 5

        if not es_fin_semana:
            carga_hora = 0.4 + 0.5 * np.exp(-((h - 13) ** 2) / 24)
        else:
            carga_hora = 0.2 + 0.1 * np.sin(2 * np.pi * h / 24)

        if es_verano and not es_fin_semana:
            carga_hora *= 1.25

        perfil_base[i] = carga_hora

    promedio_actual = np.mean(perfil_base)
    if promedio_actual > 0:
        perfil_base = perfil_base * (factor_planta / promedio_actual)

    perfil_base = (perfil_base / np.max(perfil_base)) * demanda_max

    np.random.seed(42)
    ruido       = np.random.normal(0, 0.05, len(tiempos))
    demanda_kw  = np.clip(perfil_base + (perfil_base * ruido), 0, demanda_max)
    demanda_kwh = demanda_kw * (15 / 60)

    return demanda_kw, demanda_kwh


def generar_demanda_historica_streger(tiempos):
    """
    Genera la curva de demanda escalando el perfil sintético diario
    para que el total kWh de cada mes coincida con los datos reales del
    recibo CFE de STREGER S.A. (mayo 2025 – mayo 2026).
    Demanda máxima limitada a 80 kW (real = 1 unidad × multiplicador 80).
    """
    mes        = tiempos.month
    dia_semana = tiempos.weekday
    hora       = tiempos.hour + tiempos.minute / 60.0

    # Perfil diario base (forma gaussiana de turno industrial)
    perfil_base = np.zeros(len(tiempos))
    for i, (h, ds) in enumerate(zip(hora, dia_semana)):
        if ds < 5:
            perfil_base[i] = 0.4 + 0.5 * np.exp(-((h - 13) ** 2) / 24)
        else:
            perfil_base[i] = 0.2 + 0.1 * np.sin(2 * np.pi * h / 24)

    demanda_kw = np.zeros(len(tiempos))

    for m_num in range(1, 13):
        mask        = (mes == m_num)
        perfil_mes  = perfil_base[mask]
        if perfil_mes.sum() == 0:
            continue

        target_kwh   = STREGER_KWH_MENSUAL[m_num]
        current_kwh  = perfil_mes.sum() * (15 / 60)
        scale_factor = target_kwh / current_kwh

        scaled = np.clip(perfil_mes * scale_factor, 0, STREGER_DEMANDA_MAX_KW)
        demanda_kw[mask] = scaled

    np.random.seed(42)
    ruido       = np.random.normal(0, 0.03, len(tiempos))
    demanda_kw  = np.clip(demanda_kw * (1 + ruido), 0, STREGER_DEMANDA_MAX_KW)
    demanda_kwh = demanda_kw * (15 / 60)

    return demanda_kw, demanda_kwh


@st.cache_data(show_spinner=False)
def obtener_datos_clima_era5(lat: float, lon: float, year: int = 2023):
    """
    Descarga temperatura ambiente (2 m) y velocidad de viento (10 m) de ERA5
    para un año completo. El archivo .nc se guarda localmente para no repetir
    la descarga. Año fijo 2023: garantiza disponibilidad completa en ERA5.
    """
    nc_filename  = f"era5_clima_{lat:.2f}_{lon:.2f}_{year}.nc"
    zip_filename = f"era5_clima_{lat:.2f}_{lon:.2f}_{year}.zip"

    if not os.path.exists(nc_filename):
        status = st.status("☁️ Descargando datos climáticos ERA5...", expanded=True)
        status.write("Conectando con el servidor de Copernicus CDS...")

        request = {
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "location": {"longitude": lon, "latitude": lat},
            "date": [f"{year}-01-01/{year}-12-31"],
            "data_format": "netcdf"
        }

        CDS_URL = "https://cds.climate.copernicus.eu/api"
        CDS_KEY = "23fa21b2-6d1d-457e-b307-683368fcaefe"
        cliente = cdsapi.Client(url=CDS_URL, key=CDS_KEY, quiet=True)

        status.write("Solicitud enviada. Esperando respuesta (puede tardar varios minutos)...")
        cliente.retrieve("reanalysis-era5-single-levels-timeseries", request, zip_filename)
        status.write("Descarga completada. Procesando archivo...")

        if zipfile.is_zipfile(zip_filename):
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                extracted = zip_ref.namelist()
                if not extracted:
                    raise ValueError("El ZIP descargado de ERA5 está vacío.")
                zip_ref.extractall(".")
                os.rename(extracted[0], nc_filename)
            os.remove(zip_filename)
        else:
            os.rename(zip_filename, nc_filename)

        status.update(label="✅ Datos ERA5 listos.", state="complete", expanded=False)

    ds = xr.open_dataset(nc_filename, engine="netcdf4")
    if 'time' in ds.dims and 'valid_time' not in ds.dims:
        ds = ds.rename({'time': 'valid_time'})

    temp_c = ds['t2m'] - 273.15
    viento = np.sqrt(ds['u10'] ** 2 + ds['v10'] ** 2)

    return temp_c, viento, ds['valid_time'].values


def simular_sistema_fv(lat, lon, alt, tz, tilt, azimuth, area, ef,
                       n_paneles, tiempos, distancia_filas, longitud_panel, num_filas,
                       coef_temp=-0.0035):
    """
    Pipeline de generación FV con sombreado por filas.
    Usa pvlib.location.Location para cálculo correcto de masa de aire y clearsky.
    """
    location = pvlib.location.Location(lat, lon, tz=tz, altitude=alt)
    sol      = location.get_solarposition(tiempos)
    clearsky = location.get_clearsky(tiempos, model='ineichen')

    irradiance_total = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=sol['apparent_zenith'],
        solar_azimuth=sol['azimuth'],
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi']
    )

    poa_original = irradiance_total['poa_global']

    # --- Modelo térmico: temperatura de celda via Faiman (ERA5) ---
    temp_c, viento_ds, tiempos_era5 = obtener_datos_clima_era5(lat, lon, year=2023)

    # Mapear tiempos de simulación (cualquier año) al año ERA5 para la interpolación
    tiempos_naive  = tiempos.tz_localize(None)
    tiempos_mapped = tiempos_naive.map(lambda t: t.replace(year=2023))

    climate_ds = xr.Dataset({'temp_c': temp_c, 'viento': viento_ds}).interp(
        valid_time=tiempos_mapped, method='linear'
    )
    temp_ambiente = climate_ds['temp_c'].ffill(dim='valid_time').bfill(dim='valid_time').values
    vel_viento    = climate_ds['viento'].ffill(dim='valid_time').bfill(dim='valid_time').values

    temp_celda = pvlib.temperature.faiman(
        poa_global=poa_original.values,
        temp_air=temp_ambiente,
        wind_speed=vel_viento
    )

    # Eficiencia real degradada por temperatura (STC = 25 °C)
    eficiencia_real = np.clip(ef * (1 + coef_temp * (temp_celda - 25)), 0, 1)

    # --- Sombreado inter-filas ---
    altura_panel = longitud_panel * np.sin(np.radians(tilt))

    shaded_frac_por_fila = pvlib.shading.shaded_fraction1d(
        solar_zenith=sol['apparent_zenith'],
        solar_azimuth=sol['azimuth'],
        axis_azimuth=azimuth - 90,
        shaded_row_rotation=tilt,
        collector_width=altura_panel,
        pitch=distancia_filas,
        axis_tilt=0,
        surface_to_axis_offset=0,
        cross_axis_slope=0
    )

    sombra_total_arreglo   = ((num_filas - 1) * shaded_frac_por_fila) / num_filas
    poa_con_sombra         = poa_original * (1 - sombra_total_arreglo)

    # Potencia con eficiencia degradada térmicamente
    potencia_sin_sombra_kw = (poa_original  * area * eficiencia_real * n_paneles) / 1000
    potencia_con_sombra_kw = (poa_con_sombra * area * eficiencia_real * n_paneles) / 1000
    energia_kwh            = potencia_con_sombra_kw * (15 / 60)

    return poa_original, potencia_con_sombra_kw, potencia_sin_sombra_kw, sombra_total_arreglo, energia_kwh, temp_celda


def render_kpi(col, label, value):
    col.markdown(f"""
<div style="background:rgba(245,158,11,0.07);border-left:4px solid #F59E0B;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">{label}</div>
    <div style="font-size:1.65rem;font-weight:700;color:#1E293B;">{value}</div>
</div>""", unsafe_allow_html=True)


def obtener_zona_horaria(lat, lon):
    tf   = TimezoneFinder()
    zona = tf.timezone_at(lng=lon, lat=lat)
    return zona if zona else 'UTC'


# =============================================================================
# SIDEBAR — SECCIÓN 1: INPUTS REACTIVOS (FUERA DEL FORM)
# =============================================================================
st.sidebar.header("🛠️ Configuración de Parámetros")

with st.sidebar.expander("1. Geolocalización", expanded=True):
    latitud  = st.number_input("Latitud (°)",    value=19.4791,  step=0.1,  format="%.4f")
    longitud = st.number_input("Longitud (°)",   value=-96.9500, step=0.1,  format="%.4f")
    altitud  = st.number_input("Altitud (msnm)", value=1210,     step=1)

zona_horaria = obtener_zona_horaria(latitud, longitud)
tilt_optimo  = calcular_tilt_optimo(latitud)

with st.sidebar.expander("2. Geometría de Instalación", expanded=True):
    st.caption(f"Ángulo óptimo estimado para lat. {latitud:.2f}°: **{tilt_optimo}°**")
    inclinacion = st.slider("Inclinación / Tilt (°)",   min_value=0, max_value=90,  value=tilt_optimo)
    azimut      = st.slider("Orientación / Azimut (°)", min_value=0, max_value=360, value=180,
                            help="180° indica orientación al Sur")

# Modo de demanda — fuera del form para que sea reactivo y afecte la sección 5
st.sidebar.markdown("---")
modo_demanda = st.sidebar.radio(
    "Fuente de datos de demanda:",
    ["Sintético", "Histórico STREGER"],
    index=0,
    help="'Histórico STREGER' usa los kWh reales del recibo CFE (mayo 2025 – mayo 2026)."
)

# =============================================================================
# SECCIÓN 1: DIAGNÓSTICO CLIMÁTICO (SIEMPRE VISIBLE, REACTIVO)
# =============================================================================
st.markdown("""
<div style="border-left:5px solid #F59E0B; padding:0.5rem 1rem;
            background:rgba(245,158,11,0.05); border-radius:0 6px 6px 0; margin-bottom:1.2rem;">
    <span style="font-size:0.7rem;font-weight:600;color:#92400E;text-transform:uppercase;letter-spacing:0.06em;">
        Siempre Visible
    </span>
    <h3 style="margin:0.15rem 0 0 0;color:#1E293B;font-size:1.15rem;">
        Sección 1 — Diagnóstico Climático
    </h3>
    <p style="color:#64748b;font-size:0.82rem;margin:0.25rem 0 0 0;">
        Recurso solar para la ubicación seleccionada. Se actualiza instantáneamente al cambiar
        coordenadas, inclinación u orientación. No requiere iniciar la simulación completa.
    </p>
</div>
""", unsafe_allow_html=True)

tiempos_dia, clearsky_dia, poa_dia, kwh_1panel = calcular_seccion1(
    latitud, longitud, altitud, zona_horaria, inclinacion, azimut
)

col_s1a, col_s1b, col_s1c = st.columns(3)
render_kpi(col_s1a, "Ángulo óptimo para esta latitud",        f"{tilt_optimo}°")
render_kpi(col_s1b, "Producción est. 1 panel (día típico)",   f"{kwh_1panel:.3f} kWh/día")
render_kpi(col_s1c, "Irradiancia pico estimada (día típico)", f"{int(poa_dia.max())} W/m²")

st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

fig_s1 = go.Figure()
fig_s1.add_trace(go.Scatter(
    x=tiempos_dia, y=clearsky_dia['ghi'],
    name='GHI — Global Horizontal', mode='lines',
    line=dict(color='#F59E0B', width=2)
))
fig_s1.add_trace(go.Scatter(
    x=tiempos_dia, y=clearsky_dia['dni'],
    name='DNI — Directa Normal', mode='lines',
    line=dict(color='#EF4444', width=2)
))
fig_s1.add_trace(go.Scatter(
    x=tiempos_dia, y=clearsky_dia['dhi'],
    name='DHI — Difusa Horizontal', mode='lines',
    line=dict(color='#3B82F6', width=2)
))
fig_s1.add_trace(go.Scatter(
    x=tiempos_dia, y=poa_dia,
    name=f'POA — Plano del Arreglo ({inclinacion}°, Az {azimut}°)', mode='lines',
    line=dict(color='#10B981', width=2.5, dash='dash')
))
fig_s1.update_layout(
    title=dict(
        text="Componentes de Irradiancia Solar — Día Típico (21 Jun 2026, Solsticio de Verano)",
        font=dict(size=13)
    ),
    xaxis_title="Hora del día",
    yaxis_title="Irradiancia (W/m²)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20),
    height=380
)
st.plotly_chart(fig_s1, use_container_width=True)

# Separador visual entre Sección 1 y simulación completa
st.markdown("<hr style='border:none;border-top:2px solid #e2e8f0;margin:2rem 0;'>", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — SIMULACIÓN COMPLETA (DENTRO DEL FORM)
# =============================================================================
st.sidebar.markdown("---")

with st.sidebar.form(key="formulario_parametros"):

    with st.sidebar.expander("3. Especificaciones de Paneles", expanded=True):
        potencia_w       = st.number_input("Potencia de un panel (W)", value=425, step=5)
        eficiencia       = st.slider("Eficiencia STC del panel (%)", min_value=10.0, max_value=30.0,
                                     value=21.8, step=0.1) / 100.0
        coef_temp        = st.number_input(
            "Coeficiente de temperatura (%/°C)",
            value=-0.35, step=0.01, format="%.2f",
            help="Pérdida de eficiencia por grado sobre 25 °C (STC). Típico: −0.35 %/°C para silicio monocristalino."
        ) / 100.0
        area_panel       = st.number_input("Área del panel (m²)", value=1.95, step=0.1)
        cantidad_paneles = st.number_input("Número total de paneles", min_value=1, value=216, step=10)

    with st.sidebar.expander("4. Configuración de Arreglo (Sombreado)", expanded=False):
        distancia_filas = st.number_input("Distancia entre filas (m)", value=1.15, step=0.1)
        longitud_panel  = st.number_input("Longitud física del panel (m)", value=1.134, step=0.1)
        num_filas       = st.number_input("Número de filas", value=9, step=1)

    with st.sidebar.expander("5. Perfil de Demanda Planta", expanded=False):
        if modo_demanda == "Histórico STREGER":
            st.info("Usando datos reales del recibo CFE de STREGER S.A. (mayo 2025 – mayo 2026).")
            st.caption(f"Demanda máxima real: **{STREGER_DEMANDA_MAX_KW} kW**  |  Consumo anual: **~{sum(STREGER_KWH_MENSUAL.values()):,} kWh**")
            demanda_maxima = STREGER_DEMANDA_MAX_KW
            factor_planta  = 0.22
        else:
            demanda_maxima = st.number_input("Demanda máxima (kW)", value=80, step=5)
            factor_planta  = st.slider("Factor de carga", min_value=0.00, max_value=1.00,
                                       value=0.60, step=0.01)

    with st.sidebar.expander("6. Parámetros Económicos", expanded=False):
        precio_kwh  = st.number_input("Precio de la energía ($ / kWh)", min_value=0.0,
                                      value=2.82, step=0.01, format="%.2f")
        tipo_moneda = st.selectbox("Divisa", ["MXN ($)", "USD ($)"], index=0)

    ejecutar_simulacion = st.form_submit_button(label="🚀 Iniciar Simulación", use_container_width=True)


# =============================================================================
# LÓGICA DE CONTROL Y RENDERIZADO (SIMULACIÓN COMPLETA)
# =============================================================================
if "df_resultados" not in st.session_state:
    st.session_state.df_resultados = None
    st.session_state.resumen       = None

if ejecutar_simulacion or st.session_state.df_resultados is None:
    with st.spinner("Procesando modelos físicos de radiación solar y demanda..."):
        tiempos = pd.date_range(
            start='2026-01-01 00:00',
            end='2026-12-31 23:45',
            freq='15min',
            tz=zona_horaria
        )

        try:
            poa, pot_fv_kw, pot_sin_sombra_kw, frac_sombra, env_fv_kwh, temp_celda = simular_sistema_fv(
                latitud, longitud, altitud, zona_horaria, inclinacion, azimut,
                area_panel, eficiencia, cantidad_paneles, tiempos,
                distancia_filas, longitud_panel, num_filas, coef_temp
            )
        except Exception as e:
            st.error(f"Error durante la descarga o procesamiento de datos climáticos ERA5: {e}")
            st.info("Revisa la clave CDS configurada o la conexión a internet e intenta de nuevo.")
            st.stop()

        # Modo de demanda: sintético o histórico STREGER
        if modo_demanda == "Histórico STREGER":
            dem_kw, dem_kwh = generar_demanda_historica_streger(tiempos)
        else:
            dem_kw, dem_kwh = generar_demanda_industrial(demanda_maxima, factor_planta, tiempos)

        df = pd.DataFrame({
            'Fecha_Hora':               tiempos,
            'Demanda_kW':               dem_kw,
            'Demanda_kWh':              dem_kwh,
            'POA_Original_W_m2':        poa,
            'Temp_Celda_C':             temp_celda,
            'Generacion_Con_Sombra_kW': pot_fv_kw,
            'Generacion_Sin_Sombra_kW': pot_sin_sombra_kw,
            'Fraccion_Sombra_Arreglo':  frac_sombra,
            'Generacion_Energia_kWh':   env_fv_kwh,
            'Excedente_Deficit_kW':     pot_fv_kw - dem_kw
        })

        df['Ahorro_Monetario'] = df['Generacion_Energia_kWh'] * precio_kwh
        df['Mes']              = df['Fecha_Hora'].dt.strftime('%m - %B')
        df['Mes_Num']          = df['Fecha_Hora'].dt.month

        ahorro_anual_total  = df['Ahorro_Monetario'].sum()
        energia_anual_total = df['Generacion_Energia_kWh'].sum()

        df_mensual = df.groupby('Mes').agg({
            'Generacion_Energia_kWh': 'sum',
            'Demanda_kWh':            'sum',
            'Ahorro_Monetario':       'sum'
        }).reset_index()

        st.session_state.energia_anual  = energia_anual_total
        st.session_state.ahorro_anual   = ahorro_anual_total
        st.session_state.df_mensual     = df_mensual
        st.session_state.divisa         = tipo_moneda.split(" ")[0]
        st.session_state.modo_demanda   = modo_demanda
        st.session_state.df_resultados  = df
        st.session_state.resumen        = {
            'energia_fv_anual':    env_fv_kwh.sum(),
            'demanda_anual':       dem_kwh.sum(),
            'potencia_pico':       pot_fv_kw.max(),
            'potencia_disponible': cantidad_paneles * potencia_w / 1000
        }

# Recuperación de datos estables
df_analisis = st.session_state.df_resultados
resumen     = st.session_state.resumen
divisa      = st.session_state.get('divisa', 'MXN')

# =============================================================================
# SECCIÓN 2 — BALANCE SOLAR VS. DEMANDA INDUSTRIAL
# =============================================================================
st.markdown("""
<div style="border-left:5px solid #3B82F6; padding:0.5rem 1rem;
            background:rgba(59,130,246,0.05); border-radius:0 6px 6px 0; margin-bottom:1.2rem;">
    <span style="font-size:0.7rem;font-weight:600;color:#1E40AF;text-transform:uppercase;letter-spacing:0.06em;">
        Requiere Simulación
    </span>
    <h3 style="margin:0.15rem 0 0 0;color:#1E293B;font-size:1.15rem;">
        Sección 2 — Balance Solar vs. Demanda Industrial
    </h3>
    <p style="color:#64748b;font-size:0.82rem;margin:0.25rem 0 0 0;">
        Configure los parámetros del sistema en el panel lateral y presione "Iniciar Simulación".
        Modo activo: <strong>{modo}</strong>
    </p>
</div>
""".format(modo=st.session_state.get('modo_demanda', modo_demanda)), unsafe_allow_html=True)

# --- KPI FILA 1: Métricas del sistema ---
st.subheader("⚡ Resumen del Sistema")
col1, col2, col3, col4 = st.columns(4)
render_kpi(col1, "Energía Solar Anual",   f"{resumen['energia_fv_anual']:,.1f} kWh")
render_kpi(col2, "Consumo Planta Anual",  f"{resumen['demanda_anual']:,.1f} kWh")
render_kpi(col3, "Potencia DC Instalada", f"{resumen['potencia_disponible']:.1f} kWp")
render_kpi(col4, "Generación FV Pico",    f"{resumen['potencia_pico']:.2f} kW")

# --- KPI FILA 2: Balance de cobertura ---
st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
col5, col6 = st.columns(2)
cobertura_pct = min((resumen['energia_fv_anual'] / resumen['demanda_anual']) * 100, 100.0)
balance_neto  = resumen['energia_fv_anual'] - resumen['demanda_anual']
balance_label = (f"+{balance_neto:,.0f} kWh excedente"
                 if balance_neto >= 0 else f"{balance_neto:,.0f} kWh déficit")
render_kpi(col5, "Cobertura Solar de la Demanda", f"{cobertura_pct:.1f}%")
render_kpi(col6, "Balance Neto Anual",            balance_label)

# --- KPI FILA 3: Impacto térmico (modelo Faiman + ERA5) ---
st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
col7, col8 = st.columns(2)
temp_celda_media   = df_analisis['Temp_Celda_C'].mean()
perdida_termica_pct = abs(coef_temp) * max(0.0, temp_celda_media - 25) * 100
col7.markdown(f"""
<div style="background:rgba(239,68,68,0.07);border-left:4px solid #EF4444;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Temperatura Media de Celda</div>
    <div style="font-size:1.65rem;font-weight:700;color:#1E293B;">{temp_celda_media:.1f} °C</div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">Modelo Faiman · datos reales ERA5</div>
</div>""", unsafe_allow_html=True)
col8.markdown(f"""
<div style="background:rgba(239,68,68,0.07);border-left:4px solid #EF4444;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Pérdida Estimada por Temperatura</div>
    <div style="font-size:1.65rem;font-weight:700;color:#dc2626;">{perdida_termica_pct:.1f}%</div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">
        coef. {coef_temp*100:.2f} %/°C · ΔT = {max(0.0, temp_celda_media-25):.1f} °C sobre STC
    </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

# --- MÓDULO EXPRESS (condicional, reactivo) ---
st.markdown("---")
usar_express = st.checkbox(
    "¿Deseas calcular el impacto económico rápido por precio de kWh? (Módulo Express)",
    value=False,
    help="Ingresa tu costo promedio por kWh para ver el gasto antes y después de instalar los paneles."
)

if usar_express:
    col_ep, col_er1, col_er2, col_er3 = st.columns([1, 1, 1, 1])
    with col_ep:
        precio_express = st.number_input(
            "Costo promedio kWh ($/kWh)",
            min_value=0.01,
            value=STREGER_PRECIO_MEDIO,
            step=0.01,
            format="%.2f",
            help=f"Precio medio histórico de STREGER: ${STREGER_PRECIO_MEDIO}/kWh (sin IVA)"
        )

    demanda_kWh_anual = resumen['demanda_anual']
    gen_kWh_anual     = resumen['energia_fv_anual']
    neto_kwh          = max(demanda_kWh_anual - gen_kWh_anual, 0)

    costo_actual      = demanda_kWh_anual * precio_express
    costo_con_paneles = neto_kwh          * precio_express
    ahorro_express    = costo_actual - costo_con_paneles

    render_kpi(col_er1, f"Gasto energético actual (año)",    f"${costo_actual:,.0f} {divisa}")
    render_kpi(col_er2, f"Gasto con sistema solar (año)",    f"${costo_con_paneles:,.0f} {divisa}")
    render_kpi(col_er3, f"Ahorro neto estimado (año)",       f"${ahorro_express:,.0f} {divisa}")

st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

# --- GRÁFICA 1: Series temporales (filtro de fechas) ---
st.subheader("📈 Generación VS. Demanda en el tiempo")
st.caption("Filtra un rango de fechas para inspeccionar la interacción entre generación solar y demanda de la planta. Datos del año 2026.")

col_f1, col_f2 = st.columns(2)
with col_f1:
    fecha_inicio = st.date_input("Fecha Inicio", value=pd.to_datetime("2026-03-15"))
with col_f2:
    fecha_fin = st.date_input("Fecha Fin", value=pd.to_datetime("2026-03-21"))

df_analisis['Fecha_Solo'] = df_analisis['Fecha_Hora'].dt.date
df_filtrado = df_analisis[
    (df_analisis['Fecha_Solo'] >= fecha_inicio) &
    (df_analisis['Fecha_Solo'] <= fecha_fin)
]

if not df_filtrado.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtrado['Fecha_Hora'], y=df_filtrado['Demanda_kW'],
        mode='lines', name='Demanda Industrial (kW)',
        line=dict(color='#3B82F6', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_filtrado['Fecha_Hora'], y=df_filtrado['Generacion_Con_Sombra_kW'],
        mode='lines', name='Generación Solar FV (kW)',
        line=dict(color='#F59E0B', width=2.5),
        fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.15)'
    ))
    fig.update_layout(
        xaxis_title="Fecha y Hora",
        yaxis_title="Potencia Eléctrica (kW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No hay datos disponibles para el rango de fechas seleccionado.")

# --- GRÁFICA 2: Histograma mensual Demanda vs. Generación ---
st.subheader("📊 Balance Mensual: Demanda vs. Generación Solar")
st.caption("Comparativa de energía consumida y generada por mes. Permite identificar los meses con mayor y menor cobertura solar.")

meses_es = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun',
            7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}

df_balance = (
    df_analisis
    .assign(Mes_Num=df_analisis['Fecha_Hora'].dt.month)
    .groupby('Mes_Num', sort=True)
    .agg(Demanda_Total_kWh=('Demanda_kWh', 'sum'),
         Generacion_Total_kWh=('Generacion_Energia_kWh', 'sum'))
    .reset_index()
)
df_balance['Mes_Label'] = df_balance['Mes_Num'].map(meses_es)

fig_balance = go.Figure()
fig_balance.add_trace(go.Bar(
    x=df_balance['Mes_Label'],
    y=df_balance['Demanda_Total_kWh'],
    name='Demanda (kWh)',
    marker_color='#3B82F6',
    text=[f"{v:,.0f}" for v in df_balance['Demanda_Total_kWh']],
    textposition='outside',
    textfont=dict(size=10)
))
fig_balance.add_trace(go.Bar(
    x=df_balance['Mes_Label'],
    y=df_balance['Generacion_Total_kWh'],
    name='Generación FV (kWh)',
    marker_color='#F59E0B',
    text=[f"{v:,.0f}" for v in df_balance['Generacion_Total_kWh']],
    textposition='outside',
    textfont=dict(size=10)
))
fig_balance.update_layout(
    barmode='group',
    xaxis_title="Mes",
    yaxis_title="Energía (kWh)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=40, b=40),
    height=420
)
st.plotly_chart(fig_balance, use_container_width=True)

# =============================================================================
# SECCIÓN 3 — INGENIERÍA DE RESILIENCIA ANTE APAGONES
# =============================================================================
st.markdown("<hr style='border:none;border-top:2px solid #e2e8f0;margin:2rem 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="border-left:5px solid #8B5CF6; padding:0.5rem 1rem;
            background:rgba(139,92,246,0.05); border-radius:0 6px 6px 0; margin-bottom:1.2rem;">
    <span style="font-size:0.7rem;font-weight:600;color:#5B21B6;text-transform:uppercase;letter-spacing:0.06em;">
        Condicional — Se activa al mover las horas de respaldo por encima de 0
    </span>
    <h3 style="margin:0.15rem 0 0 0;color:#1E293B;font-size:1.15rem;">
        Sección 3 — Ingeniería de Resiliencia ante Apagones
    </h3>
    <p style="color:#64748b;font-size:0.82rem;margin:0.25rem 0 0 0;">
        Dimensionamiento del sistema de respaldo BESS (LiFePO₄) para proteger la carga crítica industrial.
        Mueve el deslizador para activar el análisis.
    </p>
</div>
""", unsafe_allow_html=True)

col_s3_inp1, col_s3_inp2 = st.columns([1, 2])
with col_s3_inp1:
    horas_respaldo = st.slider(
        "Horas de respaldo deseadas",
        min_value=0, max_value=24, value=0, step=1,
        help="Horas que el BESS debe mantener la carga crítica durante un apagón."
    )
with col_s3_inp2:
    if horas_respaldo > 0:
        carga_critica_kw = st.number_input(
            "Potencia de la carga crítica (kW)",
            min_value=1.0,
            max_value=float(STREGER_DEMANDA_MAX_KW),
            value=30.0,
            step=1.0,
            help="Potencia total de los equipos críticos que deben permanecer en operación durante el corte."
        )
    else:
        st.info(
            "Mueve el deslizador de horas hacia la derecha para dimensionar el sistema BESS.",
            icon="💡"
        )

if horas_respaldo > 0:
    # Parámetros de tecnología LiFePO4
    DOD_LIFEPO4      = 0.80   # Profundidad de descarga segura
    CAPACIDAD_MODULO = 5.0    # kWh por módulo estándar (ej. Pylontech US5000)
    CICLOS_MIN       = 4_000
    CICLOS_MAX       = 5_000

    capacidad_requerida_kwh = horas_respaldo * carga_critica_kw
    capacidad_instalada_kwh = capacidad_requerida_kwh / DOD_LIFEPO4
    n_modulos               = int(np.ceil(capacidad_instalada_kwh / CAPACIDAD_MODULO))
    capacidad_total_kwh     = n_modulos * CAPACIDAD_MODULO
    vida_min_anos           = round(CICLOS_MIN / 365)
    vida_max_anos           = round(CICLOS_MAX / 365)
    pct_carga_critica       = min((carga_critica_kw / resumen['potencia_pico']) * 100, 100.0)

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    col_b1.markdown(f"""
<div style="background:rgba(139,92,246,0.07);border-left:4px solid #8B5CF6;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Capacidad Requerida</div>
    <div style="font-size:1.65rem;font-weight:700;color:#1E293B;">{capacidad_requerida_kwh:.1f} kWh</div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">{horas_respaldo}h × {carga_critica_kw:.0f} kW</div>
</div>""", unsafe_allow_html=True)

    col_b2.markdown(f"""
<div style="background:rgba(139,92,246,0.07);border-left:4px solid #8B5CF6;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Capacidad Instalada (DoD 80%)</div>
    <div style="font-size:1.65rem;font-weight:700;color:#1E293B;">{capacidad_instalada_kwh:.1f} kWh</div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">Protege la vida útil de las celdas</div>
</div>""", unsafe_allow_html=True)

    col_b3.markdown(f"""
<div style="background:rgba(139,92,246,0.07);border-left:4px solid #8B5CF6;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Módulos LiFePO₄ Necesarios</div>
    <div style="font-size:1.65rem;font-weight:700;color:#1E293B;">{n_modulos} módulos</div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">{CAPACIDAD_MODULO:.0f} kWh c/u → {capacidad_total_kwh:.0f} kWh instalados</div>
</div>""", unsafe_allow_html=True)

    col_b4.markdown(f"""
<div style="background:rgba(139,92,246,0.07);border-left:4px solid #8B5CF6;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Vida Útil Estimada</div>
    <div style="font-size:1.65rem;font-weight:700;color:#1E293B;">{vida_min_anos}–{vida_max_anos} años</div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">{CICLOS_MIN:,}–{CICLOS_MAX:,} ciclos garantizados</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:1rem 1.4rem;">
    <div style="font-weight:600;color:#1E293B;margin-bottom:0.6rem;">
        📋 Propuesta de Sistema BESS — Tecnología LiFePO₄
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.85rem;color:#475569;">
        <div>• <strong>Tecnología:</strong> Litio Ferro-Fosfato (LiFePO₄)</div>
        <div>• <strong>Profundidad de descarga segura:</strong> 80%</div>
        <div>• <strong>Ciclos garantizados:</strong> {CICLOS_MIN:,} – {CICLOS_MAX:,} ciclos</div>
        <div>• <strong>Vida útil (1 ciclo/día):</strong> {vida_min_anos} – {vida_max_anos} años</div>
        <div>• <strong>Módulos propuestos:</strong> {n_modulos} × {CAPACIDAD_MODULO:.0f} kWh c/u</div>
        <div>• <strong>Capacidad nominal total:</strong> {capacidad_total_kwh:.0f} kWh instalados</div>
        <div>• <strong>Carga crítica protegida:</strong> {carga_critica_kw:.0f} kW por {horas_respaldo} h</div>
        <div>• <strong>Carga crítica vs. pico de planta:</strong> {pct_carga_critica:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# ANÁLISIS ECONÓMICO Y DE AHORROS
# =============================================================================
st.subheader("💰 Análisis de Impacto Económico")

col_econ1, col_econ2, col_econ3 = st.columns(3)
ahorro_promedio_mes = st.session_state.ahorro_anual / 12
render_kpi(col_econ1, "Tarifa Eléctrica Base",           f"{precio_kwh:.2f} {divisa}/kWh")
render_kpi(col_econ2, "Ahorro Económico Anual Estimado", f"{st.session_state.ahorro_anual:,.2f} {divisa}")
render_kpi(col_econ3, "Ahorro Mensual Promedio",         f"{ahorro_promedio_mes:,.2f} {divisa}")

st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

with st.expander("📊 Ver desglose y tabla de ahorros mes por mes", expanded=False):
    st.markdown("Retorno económico mensualizado calculado de forma directa:")

    tabla_formateada = st.session_state.df_mensual[
        ['Mes', 'Generacion_Energia_kWh', 'Ahorro_Monetario']
    ].copy()
    tabla_formateada.columns = [
        'Mes de Simulación',
        'Energía Generada (kWh)',
        f'Ahorro Estimado ({divisa})'
    ]
    st.dataframe(
        tabla_formateada.style.format({
            'Energía Generada (kWh)':                    '{:,.2f}',
            f'Ahorro Estimado ({divisa})':               '${:,.2f}'
        }),
        use_container_width=True
    )

st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# SECCIÓN 4 — EVALUACIÓN FINANCIERA DE CONTINUIDAD DE NEGOCIO
# =============================================================================
st.markdown("<hr style='border:none;border-top:2px solid #e2e8f0;margin:2rem 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="border-left:5px solid #10B981; padding:0.5rem 1rem;
            background:rgba(16,185,129,0.05); border-radius:0 6px 6px 0; margin-bottom:1.2rem;">
    <span style="font-size:0.7rem;font-weight:600;color:#065F46;text-transform:uppercase;letter-spacing:0.06em;">
        Condicional — Desbloquear para análisis de rentabilidad avanzado
    </span>
    <h3 style="margin:0.15rem 0 0 0;color:#1E293B;font-size:1.15rem;">
        Sección 4 — Evaluación Financiera de Continuidad de Negocio
    </h3>
    <p style="color:#64748b;font-size:0.82rem;margin:0.25rem 0 0 0;">
        ROI, payback y simulación de cashflow a 25 años. Ingresa las cotizaciones reales del sistema.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("🔓 Activar análisis de rentabilidad avanzado", expanded=False):

    st.markdown("##### Parámetros de Inversión y Continuidad")
    col_inv1, col_inv2, col_inv3 = st.columns(3)

    with col_inv1:
        default_solar = round(resumen['potencia_disponible'] * 15_000, -4)
        costo_sistema_solar = st.number_input(
            "Inversión total sistema solar (MXN)",
            min_value=0.0,
            value=float(default_solar),
            step=10_000.0,
            format="%.0f",
            help="Costo total instalado: paneles + inversor + montaje + mano de obra. Referencia: ~$15,000 MXN/kWp."
        )

    with col_inv2:
        reparaciones_anuales = st.number_input(
            "Costo anual por apagones (MXN/año)",
            min_value=0.0,
            value=150_000.0,
            step=10_000.0,
            format="%.0f",
            help="Suma histórica anual de reparaciones, tiempos muertos y pérdidas de producción por cortes eléctricos."
        )

    with col_inv3:
        default_bess_cost = float(n_modulos * 18_000) if horas_respaldo > 0 else 0.0
        costo_bess = st.number_input(
            "Cotización del banco BESS (MXN)",
            min_value=0.0,
            value=default_bess_cost,
            step=10_000.0,
            format="%.0f",
            help="Cotización real con proveedor. Referencia: ~$18,000 MXN por módulo LiFePO₄ de 5 kWh."
        )

    tasa_descuento = st.slider(
        "Tasa de descuento anual (%) — para cálculo de VPN",
        min_value=0.0, max_value=25.0, value=8.0, step=0.5,
        help="Usa la TIIE o tu costo de capital. A 0% equivale al payback simple."
    ) / 100.0

    # ── Cálculos ──────────────────────────────────────────────────────────────
    VIDA_UTIL        = 25
    ahorro_e_anual   = st.session_state.ahorro_anual
    ahorro_bess      = reparaciones_anuales if costo_bess > 0 else 0.0
    inversion_total  = costo_sistema_solar + costo_bess

    anos = list(range(0, VIDA_UTIL + 1))

    # Flujos anuales discretos
    flujo_solar  = [-costo_sistema_solar]  + [ahorro_e_anual] * VIDA_UTIL
    flujo_combo  = [-inversion_total]      + [ahorro_e_anual + ahorro_bess] * VIDA_UTIL

    # Cashflow acumulado
    cf_solar_acum = list(np.cumsum(flujo_solar))
    cf_combo_acum = list(np.cumsum(flujo_combo))

    # Payback simple
    payback_solar_a = next((i for i, v in enumerate(cf_solar_acum) if v >= 0), None)
    payback_combo_a = next((i for i, v in enumerate(cf_combo_acum) if v >= 0), None)

    # VPN
    vpn_solar = sum(v / (1 + tasa_descuento) ** t for t, v in enumerate(flujo_solar))
    vpn_combo = sum(v / (1 + tasa_descuento) ** t for t, v in enumerate(flujo_combo))

    # ── KPIs ──────────────────────────────────────────────────────────────────
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    pb_solar_str = f"{payback_solar_a} años" if payback_solar_a else f"> {VIDA_UTIL} años"
    pb_combo_str = f"{payback_combo_a} años" if payback_combo_a else f"> {VIDA_UTIL} años"

    col_k1.markdown(f"""
<div style="background:rgba(16,185,129,0.07);border-left:4px solid #10B981;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Inversión Total del Proyecto</div>
    <div style="font-size:1.5rem;font-weight:700;color:#1E293B;">${inversion_total:,.0f}</div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">Solar + BESS (MXN)</div>
</div>""", unsafe_allow_html=True)

    col_k2.markdown(f"""
<div style="background:rgba(16,185,129,0.07);border-left:4px solid #10B981;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Payback Solar (simple)</div>
    <div style="font-size:1.5rem;font-weight:700;color:#1E293B;">{pb_solar_str}</div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">Ahorro: ${ahorro_e_anual:,.0f} MXN/año</div>
</div>""", unsafe_allow_html=True)

    col_k3.markdown(f"""
<div style="background:rgba(16,185,129,0.07);border-left:4px solid #10B981;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">Payback Solar + BESS</div>
    <div style="font-size:1.5rem;font-weight:700;color:#1E293B;">{pb_combo_str}</div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">Ahorro total: ${ahorro_e_anual + ahorro_bess:,.0f} MXN/año</div>
</div>""", unsafe_allow_html=True)

    col_k4.markdown(f"""
<div style="background:rgba(16,185,129,0.07);border-left:4px solid #10B981;
            border-radius:6px;padding:14px 18px;height:100%;">
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">VPN del Proyecto (tasa {tasa_descuento*100:.1f}%)</div>
    <div style="font-size:1.5rem;font-weight:700;color:{'#16a34a' if vpn_combo >= 0 else '#dc2626'}">
        ${vpn_combo:,.0f}
    </div>
    <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">
        {'Solar + BESS · Rentable ✓' if costo_bess > 0 and vpn_combo >= 0 else 'Solo Solar · Rentable ✓' if vpn_combo >= 0 else 'Revisar supuestos ✗'}
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Gráfica de Cashflow ────────────────────────────────────────────────────
    fig_cf = go.Figure()

    # Barras de flujo anual — escenario solo solar
    colores_barras = ['#ef4444' if v < 0 else 'rgba(245,158,11,0.65)' for v in flujo_solar]
    fig_cf.add_trace(go.Bar(
        x=anos,
        y=[v / 1_000 for v in flujo_solar],
        name='Flujo Anual Solar',
        marker_color=colores_barras,
        showlegend=True
    ))

    # Barras incrementales del BESS (apiladas sobre las de solar)
    if costo_bess > 0:
        flujo_bess_incremental = [-costo_bess] + [ahorro_bess] * VIDA_UTIL
        colores_bess = ['#7c3aed' if v < 0 else 'rgba(139,92,246,0.65)' for v in flujo_bess_incremental]
        fig_cf.add_trace(go.Bar(
            x=anos,
            y=[v / 1_000 for v in flujo_bess_incremental],
            name='Flujo Incremental BESS',
            marker_color=colores_bess,
        ))

    # Líneas de cashflow acumulado (eje secundario)
    fig_cf.add_trace(go.Scatter(
        x=anos,
        y=[v / 1_000 for v in cf_solar_acum],
        name='Acumulado Solar',
        mode='lines+markers',
        marker=dict(size=4),
        line=dict(color='#d97706', width=2.5),
        yaxis='y2'
    ))

    if costo_bess > 0:
        fig_cf.add_trace(go.Scatter(
            x=anos,
            y=[v / 1_000 for v in cf_combo_acum],
            name='Acumulado Solar + BESS',
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(color='#7c3aed', width=2.5),
            yaxis='y2'
        ))

    # Línea de equilibrio en eje secundario
    fig_cf.add_hline(
        y=0, yref='y2',
        line_dash='dash', line_color='#94a3b8', line_width=1.5,
        annotation_text='Punto de equilibrio', annotation_position='top right'
    )

    fig_cf.update_layout(
        title=dict(text=f"Histograma de Flujo de Caja — Proyección a {VIDA_UTIL} Años (miles MXN)", font=dict(size=13)),
        barmode='relative',
        xaxis=dict(title="Año", dtick=2),
        yaxis=dict(title="Flujo Anual (miles MXN)", side='left'),
        yaxis2=dict(title="Cashflow Acumulado (miles MXN)", overlaying='y', side='right', showgrid=False),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=60, t=60, b=20),
        height=460
    )
    st.plotly_chart(fig_cf, use_container_width=True)

    # ── Trade-off Tecnológico ─────────────────────────────────────────────────
    st.markdown("##### ⚖️ Trade-off Tecnológico")

    energia_fv_vida     = ahorro_e_anual / precio_kwh * VIDA_UTIL  # kWh totales generados en vida útil
    costo_kwh_solar     = costo_sistema_solar / energia_fv_vida if energia_fv_vida > 0 else 0
    costo_kwh_adicional = (potencia_w / 1000 * 15_000) / \
                          (kwh_1panel * 365 * VIDA_UTIL) if kwh_1panel > 0 else 0

    if costo_bess > 0 and ahorro_bess > 0:
        amort_bess_anual = costo_bess / (vida_min_anos if horas_respaldo > 0 else VIDA_UTIL)
        rentable_bess    = ahorro_bess > amort_bess_anual
        bess_ratio       = ahorro_bess / amort_bess_anual if amort_bess_anual > 0 else 0
        bess_color       = '#16a34a' if rentable_bess else '#dc2626'
        bess_texto       = (f"✅ El BESS se paga solo con los ahorros por apagones: "
                            f"${ahorro_bess:,.0f} MXN/año ahorrados vs ${amort_bess_anual:,.0f} MXN/año amortización "
                            f"(ratio {bess_ratio:.1f}×).")  \
                           if rentable_bess else \
                           (f"⚠️ El BESS no se justifica únicamente por los ahorros en reparaciones: "
                            f"${ahorro_bess:,.0f} MXN/año < ${amort_bess_anual:,.0f} MXN/año amortización. "
                            f"Considerar otros beneficios (continuidad operativa, seguro implícito).")
    else:
        bess_color = '#64748b'
        bess_texto = "Ingresa el costo del BESS y las reparaciones anuales para evaluar su rentabilidad."

    st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:1rem 1.4rem;margin-top:0.5rem;">
    <div style="margin-bottom:0.7rem;font-size:0.88rem;color:#1E293B;">
        <strong>¿Más paneles o sistema de enfriamiento activo?</strong><br>
        Costo del kWh solar a lo largo de la vida útil: <strong>${costo_kwh_solar:.2f} MXN/kWh</strong>.
        Agregar paneles adicionales tiene un costo marginal similar y no requiere mantenimiento activo,
        por lo que añadir módulos FV es generalmente preferible a instalar enfriamiento forzado
        (que añade consumo eléctrico propio y mantenimiento).
    </div>
    <div style="font-size:0.88rem;color:{bess_color};">
        <strong>¿Paneles FV o BESS como prioridad?</strong><br>
        {bess_texto}
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# MÓDULO DE ANÁLISIS AVANZADO
# =============================================================================
st.subheader("📊 Módulo de Análisis Avanzado del Proyecto")
st.markdown("Utiliza el siguiente menú para evaluar parámetros específicos del comportamiento del sistema.")

opcion_grafica = st.selectbox(
    "Selecciona los datos que deseas analizar en la gráfica:",
    [
        "Comparativa: Generación Con Sombra vs. Sin Sombra",
        "Pérdidas: Fracción de Sombra Total del Arreglo",
        "Térmica: Temperatura de Celda en el Tiempo",
        "Financiero: Ahorro Económico Mensualizado",
        "Recurso Solar: Matriz de Irradiancia Promedio Horaria"
    ]
)

_opciones_con_filtro = [
    "Comparativa: Generación Con Sombra vs. Sin Sombra",
    "Pérdidas: Fracción de Sombra Total del Arreglo",
    "Térmica: Temperatura de Celda en el Tiempo",
]
if opcion_grafica in _opciones_con_filtro:
    st.caption("Se aplica el rango de fechas seleccionado en la sección anterior.")
else:
    st.info("Esta vista muestra datos anuales completos. El filtro de fechas no aplica aquí.", icon="ℹ️")

df_filtrado_avanzado = df_analisis[
    (df_analisis['Fecha_Solo'] >= fecha_inicio) &
    (df_analisis['Fecha_Solo'] <= fecha_fin)
]

if not df_filtrado_avanzado.empty:
    fig_avanzada = go.Figure()

    if opcion_grafica == "Comparativa: Generación Con Sombra vs. Sin Sombra":
        fig_avanzada.add_trace(go.Scatter(
            x=df_filtrado_avanzado['Fecha_Hora'], y=df_filtrado_avanzado['Generacion_Sin_Sombra_kW'],
            mode='lines', name='Generación Ideal (Sin Sombra)',
            line=dict(color='#2CA02C', width=2, dash='dash')
        ))
        fig_avanzada.add_trace(go.Scatter(
            x=df_filtrado_avanzado['Fecha_Hora'], y=df_filtrado_avanzado['Generacion_Con_Sombra_kW'],
            mode='lines', name='Generación Real (Con Sombra)',
            line=dict(color='#F59E0B', width=2.5),
            fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.1)'
        ))
        fig_avanzada.update_layout(xaxis_title="Fecha y Hora", yaxis_title="Potencia Eléctrica (kW)")

    elif opcion_grafica == "Pérdidas: Fracción de Sombra Total del Arreglo":
        fig_avanzada.add_trace(go.Scatter(
            x=df_filtrado_avanzado['Fecha_Hora'],
            y=df_filtrado_avanzado['Fraccion_Sombra_Arreglo'] * 100,
            mode='lines', name='Área Sombreada del Arreglo (%)',
            line=dict(color='#7F7F7F', width=2),
            fill='tozeroy', fillcolor='rgba(127, 127, 127, 0.2)'
        ))
        fig_avanzada.update_layout(
            xaxis_title="Fecha y Hora",
            yaxis_title="Porcentaje de Sombra (%)",
            yaxis=dict(range=[0, 105])
        )

    elif opcion_grafica == "Térmica: Temperatura de Celda en el Tiempo":
        fig_avanzada.add_trace(go.Scatter(
            x=df_filtrado_avanzado['Fecha_Hora'],
            y=df_filtrado_avanzado['Temp_Celda_C'],
            mode='lines',
            name='Temperatura de Celda (°C)',
            line=dict(color='#EF4444', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(239,68,68,0.08)'
        ))
        fig_avanzada.add_hline(
            y=25, line_dash='dash', line_color='#94a3b8', line_width=1.2,
            annotation_text='STC (25 °C)', annotation_position='top right'
        )
        fig_avanzada.add_hline(
            y=temp_celda_media, line_dash='dot', line_color='#dc2626', line_width=1.2,
            annotation_text=f'Media: {temp_celda_media:.1f} °C', annotation_position='bottom right'
        )
        fig_avanzada.update_layout(
            xaxis_title="Fecha y Hora",
            yaxis_title="Temperatura de Celda (°C)",
            yaxis=dict(range=[0, max(df_filtrado_avanzado['Temp_Celda_C'].max() + 5, 60)])
        )

    elif opcion_grafica == "Financiero: Ahorro Económico Mensualizado":
        fig_avanzada.add_trace(go.Bar(
            x=st.session_state.df_mensual['Mes'],
            y=st.session_state.df_mensual['Ahorro_Monetario'],
            name=f'Ahorro ({divisa})',
            marker_color='#2CA02C',
            text=[f"${x:,.0f}" for x in st.session_state.df_mensual['Ahorro_Monetario']],
            textposition='auto',
        ))
        fig_avanzada.update_layout(
            xaxis_title="Mes",
            yaxis_title=f"Ahorro acumulado ({divisa})"
        )

    elif opcion_grafica == "Recurso Solar: Matriz de Irradiancia Promedio Horaria":
        df_completo            = df_analisis.copy()
        df_completo['Hora_Str'] = df_completo['Fecha_Hora'].dt.strftime('%H:00')
        df_completo['Mes_Num']  = df_completo['Fecha_Hora'].dt.month
        df_completo['Mes_Str']  = df_completo['Mes_Num'].map(meses_es)

        df_matriz = df_completo.groupby(
            ['Hora_Str', 'Mes_Num', 'Mes_Str']
        )['POA_Original_W_m2'].mean().reset_index()

        df_pivot         = df_matriz.pivot(index='Hora_Str', columns='Mes_Num', values='POA_Original_W_m2')
        nombres_columnas = [meses_es[m] for m in sorted(df_pivot.columns)]

        fig_avanzada = go.Figure(data=go.Heatmap(
            z=df_pivot.values,
            x=nombres_columnas,
            y=df_pivot.index,
            colorscale='Jet',
            colorbar=dict(title="Irradiancia (W/m²)"),
            hovertemplate="Mes: %{x}<br>Hora: %{y}<br>Irradiancia Promedio: %{z:.1f} W/m²<extra></extra>",
            text=np.round(df_pivot.values, 0),
            texttemplate="%{text}",
            textfont=dict(size=9, color="black")
        ))
        fig_avanzada.update_layout(
            xaxis=dict(title="Meses", side="top"),
            yaxis=dict(title="Hora del Día", autorange="reversed"),
            height=600,
            margin=dict(l=40, r=40, t=80, b=40)
        )

    fig_avanzada.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_avanzada, use_container_width=True)

# --- SECCIÓN DE EXPORTACIÓN Y DESCARGAS ---
st.subheader("💾 Descarga de Datos Estructurados")

csv_data = df_analisis.drop(columns=['Fecha_Solo']).to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Descargar Series Temporales Completas (CSV)",
    data=csv_data,
    file_name="generacion_y_demanda_anual.csv",
    mime="text/csv",
    use_container_width=True
)
