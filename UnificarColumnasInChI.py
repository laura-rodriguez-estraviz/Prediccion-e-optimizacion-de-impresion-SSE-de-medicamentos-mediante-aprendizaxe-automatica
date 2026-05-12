import argparse
from collections import defaultdict, OrderedDict
import requests
import pandas as pd


def _get(url, params=None, timeout=10):
    try:
        # Fai unha petición GET
        r = requests.get(url, params=params, timeout=timeout)

        # Se a resposta é correcta devolve o resultado
        if r.status_code == 200:
            return r

    # Se hai calquera erro ignórao
    except Exception:
        pass

    return None


def lookup_pubchem(name: str):
    # Codifica o nome para usalo nunha URL
    encoded = requests.utils.quote(name)

    # Busca o composto en PubChem Compound
    r = _get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/InChIKey/JSON"
    )

    if r:

        # Extrae as propiedades do JSON
        props = r.json().get("PropertyTable", {}).get("Properties", [])

        if props:
            # Devolve o InChIKey
            return props[0].get("InChIKey"), "PubChem Compound"

    return None, "PubChem Compound"


def lookup_pubchem_substance(name: str):
    encoded = requests.utils.quote(name)

    # Busca na base PubChem Substance
    r = _get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/name/{encoded}/property/InChIKey/JSON"
    )

    if r:

        props = r.json().get("PropertyTable", {}).get("Properties", [])

        if props:
            return props[0].get("InChIKey"), "PubChem Substance"

    return None, "PubChem Substance"


def lookup_chembl(name: str):
    # Busca o composto en ChEMBL
    r = _get(
        "https://www.ebi.ac.uk/chembl/api/data/molecule/search",

        params={
            "q": name,
            "format": "json",
            "limit": 1
        }
    )

    if r:

        mols = r.json().get("molecules", [])

        if mols:

            # Extrae o InChIKey da estrutura molecular
            ikey = (
                    mols[0].get("molecule_structures") or {}
            ).get("standard_inchi_key")

            if ikey:
                return ikey, "ChEMBL"

    return None, "ChEMBL"


def get_inchikey(name: str):
    # Proba varias fontes ata atopar o composto
    for fn in (
            lookup_pubchem,
            lookup_pubchem_substance,
            lookup_chembl
    ):

        ikey, src = fn(name)

        # Se atopa o InChIKey devolve o resultado
        if ikey:
            return ikey, src

    # Se non aparece en ningunha API
    return None, "NOT FOUND"


def merge_columns(df, col_map):
    # Agrupa columnas equivalentes
    groups = defaultdict(list)

    for c, canon in col_map.items():
        groups[canon].append(c)

    out = OrderedDict()

    # Garda columnas xa usadas
    used = set()

    for col in df.columns:

        # Nome canónico da columna
        canon = col_map.get(col, col)

        # Evita repetir columnas
        if canon in used:
            continue

        used.add(canon)

        cols = groups[canon]

        # Se só hai unha columna déixaa igual
        if len(cols) == 1:

            out[canon] = df[col]

        else:

            # Colle todas as columnas equivalentes
            sub = df[cols]

            # Intenta converter os datos a números
            num = sub.apply(pd.to_numeric, errors="coerce")

            # suma os valores
            out[canon] = (
                num.sum(axis=1, min_count=1)
            )

    return pd.DataFrame(out)


def merge_keyword_columns(df, keyword, new_col):
    # Busca columnas que conteñan a keyword
    cols = [
        c for c in df.columns
        if keyword.lower() in c.lower()
    ]

    # Se non hai columnas coincidentes
    if not cols:
        df[new_col] = None

        return df

    # Converte os valores a numéricos
    num = df[cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Suma por filas
    df[new_col] = num.sum(
        axis=1,
        min_count=1
    )

    return df


def main():
    # Configura argumentos de terminal
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="DatosLimpios.csv"
    )

    ap.add_argument(
        "--output",
        default="DatosUnificados.csv"
    )

    ap.add_argument(
        "--stop-at",
        default="xanthan gum"
    )

    args = ap.parse_args()

    # Le o CSV
    df = pd.read_csv(args.input)

    # Lista de columnas
    cols = list(df.columns)

    # Busca ata que columna procesar ingredientes
    stop_idx = next(
        (
            i for i, c in enumerate(cols)
            if c.lower() == args.stop_at.lower()
        ),
        len(cols)
    )

    # Columnas consideradas ingredientes
    ingredient_cols = cols[:stop_idx + 1]

    # Garda resultados: columna -> (InChIKey, fonte)
    results = {}

    # Garda materiais por fonte
    sources = defaultdict(list)

    print("Buscando materiais...")

    # Busca cada ingrediente nas APIs
    for i, col in enumerate(ingredient_cols):
        ikey, src = get_inchikey(col)
        results[col] = (ikey, src)
        sources[src].append(col)
        print(f"{i + 1}/{len(ingredient_cols)} {col} → {src}")

    # Agrupa columnas co mesmo InChIKey
    groups = defaultdict(list)

    for k, (ikey, _) in results.items():
        if ikey:
            groups[ikey].append(k)

    print("\n==============================")
    print("GRUPOS FORMADOS")
    print("==============================")

    # Mostra grupos con máis dunha columna
    for i, (ikey, grp) in enumerate(groups.items(), 1):

        if len(grp) > 1:
            print(f"\nGrupo {i} — {ikey}")
            for c in grp:
                print(f"  - {c}")

    print("\n==============================")
    print("RESUME POR ORIXE")
    print("==============================")

    # Mostra cantos materiais atopou cada API
    for src, items in sources.items():
        print(f"\n{src}: {len(items)} materiais")

    # Mapa: columna -> representante do grupo
    col_map = {
        col:
            groups[ikey][0]
            if (
                    (ikey := results.get(col, (None,))[0])
                    and ikey in groups
            )
            else col
        for col in cols
    }

    original_count = len(df.columns)

    # Fusiona columnas equivalentes
    merged = merge_columns(df, col_map)

    # Fusiona columnas relacionadas con flavouring
    merged = merge_keyword_columns(merged, "flavouring", "Flavouring mode")

    merged = merge_keyword_columns(merged, "flavoring", "Flavouring mode")

    # Fusiona columnas bitter block
    merged = merge_keyword_columns(merged, "bitter bloc", "Bitter block")
    merged = merge_keyword_columns(merged, "bitter-bloc", "Bitter block")

    # Fusiona columnas colorant
    merged = merge_keyword_columns(merged, "colorant", "Colorant")

    merged = merge_keyword_columns(merged, "colourant", "Colorant")

    print("\n==============================")
    print("IMPACTO")
    print("==============================")

    # Estatísticas finais
    print(f"Columnas orixinais: {original_count}")
    print(f"Columnas finais:    {len(merged.columns)}")

    print(
        f"Redución:           "
        f"{original_count - len(merged.columns)}"
    )

    # Lista materiais non identificados
    unresolved = [
        k for k, (ikey, _) in results.items()
        if not ikey
    ]

    print("\n==============================")
    print("NON IDENTIFICADOS")
    print("==============================")

    print(f"{len(unresolved)} materiais non atopados")

    for u in unresolved:
        print(f"  - {u}")

    # Garda o CSV final
    merged.to_csv(args.output, index=False)

    print("\nGardado:", args.output)


if __name__ == "__main__":
    # Executa o programa principal
    main()
