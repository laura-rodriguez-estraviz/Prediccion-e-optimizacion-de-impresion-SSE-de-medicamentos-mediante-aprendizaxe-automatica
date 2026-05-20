from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

#Lectura dos datos
df = pd.read_csv("DatosLimpiados.csv")

#Normalización dos materiais para que en vez de cantidades haxa proporcións
all_cols = list(df.columns) #Lista cos nomes das columnas
lastMaterial_idx_raw = next(i for i, c in enumerate(all_cols) if c.strip().lower() == "xanthan gum") #Número da columna correspondente a xanthan gum
ingredient_cols = all_cols[:lastMaterial_idx_raw + 1] #Columnas dos materiais (hasta 'xanthan gum' inclusive)
row_sums = df[ingredient_cols].sum(axis=1) #Calcular suma por fila dos materiais
mask_nonzero = row_sums > 0 # Filas con suma > 0
df.loc[mask_nonzero, ingredient_cols] = (df.loc[mask_nonzero, ingredient_cols].div(row_sums[mask_nonzero], axis=0)) #Normalizo dividindo cada elemento pola suma da fila sempre que esta sume máis de 0

#División en conxuntos de adestramento e test
n_total = len(df)
n_train = int(np.floor(n_total * 0.8))
n_test  = n_total - n_train
train_df = df.iloc[:n_train].copy()
test_df  = df.iloc[n_train:].copy()
print(f"  Instancias totais:      {n_total}")
print(f"  Adestramento (80%):      {n_train}")
print(f"  Test  (20%):      {n_test}")

# Eliminación dos materiais con menos de 0.07 de aparición (aproximadamente menos de 50 aparicións)
target_cols   = ["printability", "temperaturaImpresion", "technology", "characteristics"]
other_cols     = ["printerBrandType", "preparationTemp",
                 "platformTemp", "relationExtrusionWeight", "drugConcentration"]
non_material_cols = target_cols + other_cols
freq = {}
for col in ingredient_cols:
    freq[col] = (train_df[col] != 0).sum()/ n_train #recorro as columnas e calculo a frecuencia coa que o material aparece (freq elem distinto de 0)
freq_series = pd.Series(freq)
kept_ingredients = freq_series[freq_series >= 0.07].index.tolist()
removed_ingredients = freq_series[freq_series < 0.07].index.tolist()
print("Eliminación dos materiais con menos de 0.07 de aparición (aproximadamente menos de 50 aparicións):")
print(f"  Materiais orixinalmente:  {len(ingredient_cols)}")
print(f"  Materiais eliminados:  {len(removed_ingredients)}")
print(f"  Materiais retidos:   {len(kept_ingredients)}")
mask_full  = (df[kept_ingredients] != 0).any(axis=1) #En cada fila comprobo se os ingredientes conservados son distintos de 0 (true--1) ou non (false--0)
print(f"  Instancias con alomenos presenza dun material retido: {mask_full.sum()} / {n_total}")
mask_train = (train_df[kept_ingredients] != 0).any(axis=1)
print(f"      Adestramento {mask_train.sum()} / {n_train}")
mask_test  = (test_df[kept_ingredients] != 0).any(axis=1)
print(f"      Test {mask_test.sum()} / {n_test}")
print("  Lista dos materiais retidos:")
for ing in kept_ingredients:
    print(f"    {ing:60s}  {freq[ing]:.3f}")
feature_cols = kept_ingredients + non_material_cols
work_df = train_df[feature_cols].copy()

#Correlación de variables

    #Non é o mesmo correlación entre variables categórica-categórica, numerica-numerica e categorica-numerica, así que debo separar
categorical_cols = ["printability", "technology", "characteristics","printerBrandType"]
numerical_cols = ["temperaturaImpresion", "preparationTemp", "platformTemp", "relationExtrusionWeight", "drugConcentration"] + kept_ingredients
print(f"  Variables numéricas  ({len(numerical_cols)}): {numerical_cols}")
print(f"  Variables categóricas ({len(categorical_cols)}): {categorical_cols}")
ingredient_set = set(kept_ingredients) #para maior rapidez ao buscar materiais ainda que solo son 16 por se se incrementan no futuro

    #Función que obten as filas válidas para calcular a correlación dado un par de columnas
def get_valid_rows(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """
    Devolve as filas de df onde:
      1. Ningún dos dous valores é NaN.
      2. Se AMBAS columnas son materiais, alomenos un ten valor != 0.
         Se só UNHA é material, esa columna debe ser != 0.
    """
    subset = df[[col_a, col_b]].dropna() # elimino todas as filas cun NaN
    if col_a in ingredient_set and col_b in ingredient_set:
        mask = (subset[col_a] != 0) | (subset[col_b] != 0)
        subset = subset[mask]
    elif col_a in ingredient_set:
        subset = subset[subset[col_a] != 0]
    elif col_b in ingredient_set:
        subset = subset[subset[col_b] != 0]
    return subset

    #Correlación variables numérica-numérica -- Pearson
pearson_r = pd.DataFrame(np.nan, index=numerical_cols, columns=numerical_cols) # Matriz onde se gardarán os coeficientes de correlación de Pearson (r), inicialízase con NaN
pearson_n = pd.DataFrame(0, index=numerical_cols, columns=numerical_cols) # Matriz para gardar o número de filas válidas usadas en cada cálculo
for i, col_a in enumerate(numerical_cols):
    for j, col_b in enumerate(numerical_cols):
        valid = get_valid_rows(work_df, col_a, col_b)
        if i == j: #No caso da diagonal a correlación dunha variable consigo mesma sempre é 1
            pearson_r.loc[col_a, col_b] = 1.0
            pearson_n.loc[col_a, col_b] = len(valid)
        elif j > i and len(valid) >= 3: #solo calculo a metade superior da matriz xa que a outra é simétrica e non podo calcular pearson cando hay menos de 3 valores
            r, p = stats.pearsonr(valid[col_a], valid[col_b])
            pearson_r.loc[col_a, col_b] = r
            pearson_n.loc[col_a, col_b] = len(valid)

    #Correlación variables categórica-categórica -- Cramer's V
def cramers_v(s_a: pd.Series, s_b: pd.Series) -> float:
    """Cramér's V con corrección de sesgo (Wicher Bergsma)."""
    ct = pd.crosstab(s_a, s_b)
    chi2, _, _, _ = stats.chi2_contingency(ct, correction=False)
    n   = ct.values.sum()
    r, k = ct.shape
    # corrección
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    k_c  = k - (k - 1) ** 2 / (n - 1)
    r_c  = r - (r - 1) ** 2 / (n - 1)
    denom = min(k_c - 1, r_c - 1)
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2 / denom)

cramer_v = pd.DataFrame(np.nan, index=categorical_cols, columns=categorical_cols) #Matriz para os coeficientes cramer
cramer_n = pd.DataFrame(0, index=categorical_cols, columns=categorical_cols) #Matriz para o numero e filas válidas de cada conxunto de variables
for i, col_a in enumerate(categorical_cols):
    for j, col_b in enumerate(categorical_cols):
        valid = get_valid_rows(work_df, col_a, col_b)
        if i == j:#Na diagonal ponse o coeficiente a 1 xa que non interesa
            cramer_v.loc[col_a, col_b] = 1.0
            cramer_n.loc[col_a, col_b] = len(valid)
        elif len(valid) >= 5 and j > i: #Solo calculo a parte superior da matriz simétrica e necesito mínimo 5 valores para poder calcular v
            v = cramers_v(valid[col_a].astype(str), valid[col_b].astype(str))
            cramer_v.loc[col_a, col_b] = v
            cramer_n.loc[col_a, col_b] = len(valid)

    #Correlación variables categórica-numérica -- Correlation Ratio eta
def correlation_ratio(numerical: pd.Series, categorical: pd.Series) -> float:
    """eta: varianza entre grupos / varianza total."""
    cat_vals = categorical.unique()
    grand_mean = numerical.mean()
    ss_between = sum(
        len(numerical[categorical == c]) * (numerical[categorical == c].mean() - grand_mean) ** 2
        for c in cat_vals
        if len(numerical[categorical == c]) > 0
    )
    ss_total = ((numerical - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return np.sqrt(ss_between / ss_total)

eta_matrix = pd.DataFrame(np.nan, index=numerical_cols, columns=categorical_cols) #Matriz para os coeficientes eta
eta_n = pd.DataFrame(0,      index=numerical_cols, columns=categorical_cols) #Matriz para o número de filas válidas usadas para o cálculo de cada coeficiente
for num_col in numerical_cols:
    for cat_col in categorical_cols:
        valid = get_valid_rows(work_df, num_col, cat_col)
        if len(valid) >= 5:#neste caso non se repiten as combinacións xa que non as fago dos mesmos dous vectores
            eta_matrix.loc[num_col, cat_col] = correlation_ratio(valid[num_col], valid[cat_col].astype(str))
            eta_n.loc[num_col, cat_col] = len(valid)

    #Almaceno os resultados en png
def make_heatmap(matrix: pd.DataFrame, title: str, ax: plt.Axes, vmin: float = -1, vmax: float = 1, cmap: str = "RdYlGn", fmt: str = ".2f", annot_size: int = 10):
    """Debuxa un heatmap."""
    mask = matrix.isnull()
    sns.heatmap(
        matrix.astype(float),
        ax         = ax,
        mask       = mask,
        vmin       = vmin,
        vmax       = vmax,
        cmap       = cmap,
        annot      = True,
        fmt        = fmt,
        annot_kws  = {"size": annot_size},
        linewidths = 0.4,
        linecolor  = "white",
        cbar_kws   = {"shrink": 0.8},
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=90, labelsize=10)
    ax.tick_params(axis="y", rotation=0,  labelsize=10)

plots = [
    (pearson_r,  "R de Pearson", -1, 1, "RdYlBu_r", "correlacion_pearson.png",(12, 9)),
    (cramer_v,   "V de Cramér", 0, 1, "YlOrRd", "correlacion_cramer_v.png",(4,3)),
    (eta_matrix, "Razón de correlación η", 0, 1, "YlOrRd", "correlacion_eta.png",(6,6)),
]
for matrix, title, vmin, vmax, cmap, filename, figsize in plots:
    fig, ax = plt.subplots(figsize=figsize)
    make_heatmap(matrix, title, ax, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

#Imprimo as variables con maior correlación
print("\nPares de variables con maior correlación (Top 15 por matriz):")

def top_pairs(matrix: pd.DataFrame, n_matrix: pd.DataFrame, label: str, n: int = 15, abs_val: bool = False):
    """
        Converte unha matriz completa nunha lista da forma:
        Var A	Var B	Coeficiente
    """
    rows = []
    cols_list = matrix.columns.tolist()
    idx_list = matrix.index.tolist()
    for i, r in enumerate(idx_list):
        for j, c in enumerate(cols_list):
            if r == c:#salto os valores de correlación dunha variable por si mesma
                continue
            val = matrix.loc[r, c] #extraigo o coeficiente de correlación
            if pd.isna(val):#salto os coeficientes NaN
                continue
            rows.append({"Var A": r, "Var B": c, label: val, "n_filas": n_matrix.loc[r, c]})
    df_pairs = pd.DataFrame(rows)
    if df_pairs.empty:
        return df_pairs
    df_pairs["_abs"] = df_pairs[label].abs() if abs_val else df_pairs[label] #Creo columna auxiliar na que basearei a orde do top
    return df_pairs.nlargest(n, "_abs").drop(columns="_abs").reset_index(drop=True) #devolvo o top utilizando a columna auxiliares

def add_zero_breakdown(df_top: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """
        Engade á lista con formato
        Var A	Var B	Coeficiente
        O número de casos nos que
        A≠0 e B=0       A=0 e B≠0       A=0         B=0
    """
    col_za = []
    col_zb = []
    col_a0b1 = []
    col_a1b0 = []
    for _, row in df_top.iterrows():
        va, vb = row["Var A"], row["Var B"]
        sub = get_valid_rows(source_df, va, vb)
        col_za.append(int((sub[va] == 0).sum()))
        col_zb.append(int((sub[vb] == 0).sum()))
        col_a0b1.append(int(((sub[va] == 0) & (sub[vb] != 0)).sum()))
        col_a1b0.append(int(((sub[va] != 0) & (sub[vb] == 0)).sum()))
    result = df_top.copy()
    result["n_zeros_A"] = col_za
    result["n_zeros_B"] = col_zb
    result["n(A=0,B!=0)"] = col_a0b1
    result["n(A!=0,B=0)"] = col_a1b0
    return result

print("\n     Pearson r")
top_p = top_pairs(pearson_r, pearson_n, "Pearson r", abs_val=True)
top_p = add_zero_breakdown(top_p, work_df)
print(top_p.to_string(index=False))
print("\n     Cramér's V")
top_c = top_pairs(cramer_v, cramer_n, "Cramer's V", abs_val=False)
print(top_c.to_string(index=False))
print("\n     Correlation Ratio eta")
top_e = top_pairs(eta_matrix, eta_n, "eta")
print(top_e.to_string(index=False))
# Exportar a CSV
top_p.to_csv("top_pearson.csv", index=False)
top_c.to_csv("top_cramer.csv", index=False)
top_e.to_csv("top_eta.csv", index=False)

#Feature importance
input_cols = other_cols + kept_ingredients
importance_results = {}
for target in target_cols:
    y = work_df[target].dropna() # elimino nulos de y
    valid_idx = y.index
    X = work_df.loc[valid_idx, input_cols].dropna() # elimino filas de x que eliminei por ser nulos en y + elimino nulos en x
    for col in input_cols:#normalizo as variables cualitativas
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    y = y.loc[X.index] # elimino filas en y que eliminei en x
    if y.dtype == object:#se target categorico
        y_enc = LabelEncoder().fit_transform(y.astype(str)) # normalizo y
        model = RandomForestClassifier(n_estimators=200, random_state=97, n_jobs=-1)
        model.fit(X, y_enc)
        model_type = "Clasificación"
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=97, n_jobs=-1)
        model.fit(X, y)
        model_type = "Regresión"
    imp = pd.Series(model.feature_importances_, index=input_cols).sort_values(ascending=False)#top featuress
    importance_results[target] = {}
    importance_results[target]["top"] = imp # de cada target se almacena o top
    importance_results[target]["n"] = len(y)
    print("Variables máis importantes na predicion de cada target (Top 10 por target):")
    print(f"\n  [{target}] ({model_type}, n={len(y)}) — Top 10:")
    print(imp.head(10).to_string())

#Gardo os resultados de feature importance xunto a unha representación da realación das variables máis importantes e as features
for target, data in importance_results.items():
    imp = data["top"]
    n = data["n"]
    # Gráfico de barras de importancia
    fig, ax = plt.subplots(figsize=(7, 3))
    top5 = imp.head(5)
    colors = plt.cm.Blues(np.linspace(0.1, 0.9, len(top5)))
    bars = ax.barh(top5.index[::-1], top5.values[::-1], color=colors[::-1])
    ax.set_title(f"{target}, n={n}", fontweight="bold", fontsize=15)
    ax.set_xlabel("Importancia (MDI)")
    ax.tick_params(axis="y", labelsize=10)
    for bar, val in zip(bars, top5.values[::-1]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=10
        )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"importancia_{target}.png", dpi=150, bbox_inches="tight")
    plt.close()
    # Relaciones coas 2 variables máis importantes de cada feature
    top_vars = imp.head(2).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(5 * 2, 3))
    for ax, feat in zip(axes, top_vars):
        sub = work_df[[target, feat]].dropna()
        sub = sub[sub[feat] != 0]
        feat_is_cat = (sub[feat].dtype == object)
        target_is_cat = (work_df[target].dtype == object)
        if not target_is_cat and not feat_is_cat:
            # Ambos numéricos - scatter
            ax.scatter(sub[feat], sub[target], alpha=0.5, s=25)
            ax.set_xlabel(feat)
            ax.set_ylabel(target)
        elif target_is_cat and not feat_is_cat:
            # Target categórico, feature numérico - boxplot por categoría de target
            categories = sorted(sub[target].astype(str).unique())
            data_by_cat = [sub[feat][sub[target].astype(str) == c].values for c in categories]
            ax.boxplot(data_by_cat, labels=categories)
            ax.set_xlabel(target)
            ax.set_ylabel(feat)
            ax.tick_params(axis="x", rotation=0)
        elif not target_is_cat and feat_is_cat:
            # Feature categórico, target numérico - boxplot por categoría de feature
            categories = sorted(sub[feat].astype(str).unique())
            data_by_cat = [sub[target][sub[feat].astype(str) == c].values for c in categories]
            ax.boxplot(data_by_cat, labels=categories)
            ax.set_xlabel(feat)
            ax.set_ylabel(target)
            ax.tick_params(axis="x", rotation=0)
        else:
            # Ambos categóricos - heatmap de contingencia
            ct = pd.crosstab(sub[target], sub[feat])
            sns.heatmap(ct, ax=ax)
    plt.tight_layout()
    plt.savefig(f"relacion_{target}.png", dpi=150, bbox_inches="tight")
    plt.close()

#Outliers
for col in numerical_cols:
    fig, ax = plt.subplots(figsize=(3, 2.8)) #plot do boxplot e dos eixos
    s = work_df[col].dropna()  # elimino nulos
    if col in ingredient_set or col == "drugConcentration":  # non utilizo os 0 para detectar outliers nestes casos xa que indican a ausencia, o cal non me interesa que sea a norma
        s = s[s != 0]
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75) #Cuartiles
    IQR = Q3 - Q1 #Rango intercuartílico
    lo_iqr, hi_iqr = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR #rango outliers
    outliers_mod = s[(s < lo_iqr) | (s > hi_iqr)] #outliers
    ax.boxplot(
        s,
        vert=True
    )
    ax.set_title(
        f"{col}\n "
        f"n={len(s)}, "
        f"número de outliers={len(outliers_mod)} ",
        fontsize=10
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_col=re.sub(r'[\\/*?:"<>|]', "_", col)
    filename = f"outlier_boxplot_{ save_col}.png"
    plt.savefig( filename, dpi=150, bbox_inches="tight")
    plt.close()

#Missing matrix
state_df = work_df.apply(lambda s: np.select([s.isna(), s == 0], [0, 1], default=2))
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(state_df.T, cmap=ListedColormap(["lightyellow", "navajowhite", "steelblue"]),
            ax=ax, cbar=False, xticklabels=50, yticklabels=True)
ax.set(xlabel="Instancia")
ax.legend(handles=[
    mpatches.Patch(color="lightyellow", label="NaN"),
    mpatches.Patch(color="navajowhite", label="0"),
    mpatches.Patch(color="steelblue", label="Valor")
], loc="upper right")
plt.tight_layout()
plt.savefig("missingMatrix.png", dpi=150, bbox_inches="tight")
plt.close()