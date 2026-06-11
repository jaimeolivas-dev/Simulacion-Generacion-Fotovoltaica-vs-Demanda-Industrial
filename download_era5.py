"""
Script de descarga única de datos climáticos ERA5.
Ejecutar UNA VEZ desde el terminal del devcontainer:

    python3 download_era5.py

Genera el archivo era5_clima_19.48_-96.95_2023.nc en el directorio del proyecto.
Después de generarlo, hacer commit al repo para que Streamlit Cloud lo use directamente.
"""
import cdsapi
import xarray as xr
import zipfile
import os

LAT      = 19.4791
LON      = -96.9500
YEAR     = 2023
CDS_URL  = "https://cds.climate.copernicus.eu/api"
CDS_KEY  = "23fa21b2-6d1d-457e-b307-683368fcaefe"

nc_filename  = f"era5_clima_{LAT:.2f}_{LON:.2f}_{YEAR}.nc"
zip_filename = f"era5_clima_{LAT:.2f}_{LON:.2f}_{YEAR}.zip"

if os.path.exists(nc_filename):
    print(f"✅ El archivo {nc_filename} ya existe. No es necesario descargarlo de nuevo.")
else:
    print(f"📡 Conectando con Copernicus CDS...")
    cliente = cdsapi.Client(url=CDS_URL, key=CDS_KEY, quiet=False)

    request = {
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "location": {"longitude": LON, "latitude": LAT},
        "date": [f"{YEAR}-01-01/{YEAR}-12-31"],
        "data_format": "netcdf"
    }

    print("📤 Solicitud enviada. Esperando respuesta del servidor (puede tardar varios minutos)...")
    cliente.retrieve("reanalysis-era5-single-levels-timeseries", request, zip_filename)
    print("📦 Descarga completada. Extrayendo archivo...")

    if zipfile.is_zipfile(zip_filename):
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            extracted = zip_ref.namelist()
            zip_ref.extractall(".")
            os.rename(extracted[0], nc_filename)
        os.remove(zip_filename)
    else:
        os.rename(zip_filename, nc_filename)

    print(f"✅ Archivo guardado: {nc_filename}")

# Verificación rápida del contenido
print("\n🔍 Verificando contenido del archivo...")
ds = xr.open_dataset(nc_filename, engine="netcdf4")
print(ds)
print(f"\n✅ Listo. Ahora ejecuta:")
print(f"   git add {nc_filename}")
print(f"   git commit -m 'Add ERA5 climate data for Coatepec 2023'")
print(f"   git push origin master")
