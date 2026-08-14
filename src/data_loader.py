import requests
import pandas as pd
from pathlib import Path


WHO_INDICATOR_URL = "https://ghoapi.azureedge.net/api/Indicator"
WHO_API_BASE_URL = "https://ghoapi.azureedge.net/api"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


DISEASE_INDICATORS = {
    "malaria": "MALARIA_TOTAL_CASES",
    "cholera": "CHOLERA_0000000001",
    "tuberculosis": "TB_e_inc_num",
    "meningitis": "MENING_2",
    "measles": "WHS4_543",
}


def _request_json(
    url: str,
    timeout: tuple[int, int] = (10, 120),
) -> dict:
    """Send a GET request and return JSON data."""

    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "AI-Global-Health-Intelligence/1.0"
        },
    )

    response.raise_for_status()

    return response.json()


def get_who_indicators() -> pd.DataFrame:
    """Retrieve the WHO indicator catalogue."""

    data = _request_json(
        WHO_INDICATOR_URL
    )

    return pd.DataFrame(
        data.get("value", [])
    )


def get_indicator_data(
    indicator_code: str,
) -> pd.DataFrame:
    """Retrieve observations for one WHO indicator."""

    url = (
        f"{WHO_API_BASE_URL}/"
        f"{indicator_code}"
    )

    data = _request_json(url)

    return pd.DataFrame(
        data.get("value", [])
    )


def _get_cache_path(
    disease: str,
) -> Path:
    """Return the local cache path for a disease."""

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return DATA_DIR / f"{disease}.csv"


def get_disease_data(
    disease: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Retrieve surveillance data for a supported disease.

    Previously downloaded data is stored locally so the
    project does not depend on the WHO server for every
    test or dashboard refresh.
    """

    disease = disease.lower().strip()

    if disease not in DISEASE_INDICATORS:
        supported = ", ".join(
            DISEASE_INDICATORS.keys()
        )

        raise ValueError(
            f"Unsupported disease: {disease}. "
            f"Supported diseases: {supported}"
        )

    cache_path = _get_cache_path(disease)

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path)

        cached["disease"] = disease

        return cached

    indicator_code = DISEASE_INDICATORS[
        disease
    ]

    df = get_indicator_data(
        indicator_code
    )

    if df.empty:
        raise ValueError(
            f"WHO returned no data for {disease}."
        )

    df["disease"] = disease

    df.to_csv(
        cache_path,
        index=False,
    )

    return df


def get_all_disease_data(
    use_cache: bool = True,
) -> pd.DataFrame:
    """Retrieve surveillance data for all supported diseases."""

    frames = []

    for disease in DISEASE_INDICATORS:

        try:
            disease_df = get_disease_data(
                disease,
                use_cache=use_cache,
            )

            if not disease_df.empty:
                frames.append(disease_df)

        except requests.RequestException as error:
            print(
                f"Could not retrieve {disease}: {error}"
            )

        except Exception as error:
            print(
                f"Could not process {disease}: {error}"
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )