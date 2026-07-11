import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Life Expectancy — Predictive Modelling", layout="wide")
st.title("Life Expectancy — Predictive Modelling & SHAP")

DATA_PATH = "life_expectancy.csv"
TARGET_COL = "Life Expectancy World Bank"

# ----------------------------------------------------------------------
# Dependency check — fail fast with a clear message instead of a crash
# halfway through the app.
# ----------------------------------------------------------------------
_missing = []
try:
    import xgboost as xgb
except ImportError:
    _missing.append("xgboost")
try:
    import shap
except ImportError:
    _missing.append("shap")
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    _missing.append("statsmodels")

if _missing:
    st.error(
        "Missing package(s): "
        + ", ".join(_missing)
        + ". Add them to requirements.txt and redeploy."
    )
    st.stop()

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf  # noqa: F401
except ImportError as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(e)


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(source):
    return pd.read_csv(source)


uploaded = st.sidebar.file_uploader("Upload life_expectancy.csv (optional)", type="csv")
try:
    raw_data = load_data(uploaded if uploaded is not None else DATA_PATH)
except FileNotFoundError:
    st.error(f"Couldn't find `{DATA_PATH}` next to the app. Upload the CSV in the sidebar to continue.")
    st.stop()

if TARGET_COL not in raw_data.columns:
    st.error(f"Expected a `{TARGET_COL}` column in the data — found: {list(raw_data.columns)}")
    st.stop()


# ----------------------------------------------------------------------
# 2. Cleaning / imputation
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Cleaning & imputing missing values...")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.drop(columns=["Corruption"], errors="ignore").copy()

    group_mean_cols = [
        "Life Expectancy World Bank",
        "Prevelance of Undernourishment",
        "Health Expenditure %",
        "Unemployment",
        "Sanitation",
        "CO2",
    ]
    has_region_year = {"Region", "Year"}.issubset(data.columns)
    for col in group_mean_cols:
        if col in data.columns and has_region_year:
            data[col] = data.groupby(["Region", "Year"])[col].transform(lambda x: x.fillna(x.mean()))

    if "Education Expenditure %" in data.columns and "Region" in data.columns:
        data["Education Expenditure %"] = data.groupby("Region")["Education Expenditure %"].transform(
            lambda x: x.sort_values().ffill().bfill()
        )

    # anything still missing (e.g. a region/year group that's all-NaN) -> overall column mean
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = data[numeric_cols].apply(lambda s: s.fillna(s.mean()))

    return data.dropna(subset=[TARGET_COL]).reset_index(drop=True)


data_drop = clean_data(raw_data)

with st.expander("Raw & cleaned data", expanded=False):
    st.write(f"{len(raw_data)} rows loaded, {len(data_drop)} kept after cleaning.")
    st.dataframe(raw_data.head())
    st.dataframe(data_drop.describe().T)


# ----------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------
ID_COLS = [c for c in ["Country Name", "Country Code"] if c in data_drop.columns]


@st.cache_data(show_spinner="Encoding features...")
def build_features(data: pd.DataFrame):
    x_raw = data.drop(columns=[TARGET_COL] + ID_COLS)
    y = data[TARGET_COL]

    num_cols = list(x_raw.select_dtypes(exclude="object").columns)
    cat_cols = list(x_raw.select_dtypes(include="object").columns)

    scaler = StandardScaler()
    scaler.fit(x_raw[num_cols])
    x_num = pd.DataFrame(scaler.transform(x_raw[num_cols]), columns=num_cols, index=x_raw.index)

    dummy_frames = []
    for c in cat_cols:
        dummy = pd.get_dummies(x_raw[c], prefix=c, dtype=int)
        drop_col = dummy.sum(axis=0).idxmax()  # drop the most common category as the reference level
        dummy_frames.append(dummy.drop(columns=[drop_col]))
    x_cat = pd.concat(dummy_frames, axis=1) if dummy_frames else pd.DataFrame(index=x_raw.index)

    x_full = pd.concat([x_num, x_cat], axis=1)
    x_full.columns = x_full.columns.astype(str)

    return x_full, y, num_cols, cat_cols, scaler, x_raw[num_cols]


X, y, num_cols, cat_cols, scaler, raw_numeric = build_features(data_drop)

if ID_COLS:
    st.caption(f"Excluded identifier column(s) from modelling: {', '.join(ID_COLS)} (still used to label results).")


# ----------------------------------------------------------------------
# 4. Feature selection: importance ranking + VIF pruning
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Ranking features & checking multicollinearity...")
def select_features(x: pd.DataFrame, y: pd.Series, top_n: int = 15, vif_threshold: float = 6.0):
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(x, y)
    importances = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False)
    top_features = importances.head(min(top_n, len(importances))).index.tolist()

    reduced = x[top_features].copy()
    while reduced.shape[1] > 1:
        with_const = sm.add_constant(reduced)
        vif = pd.Series(
            [variance_inflation_factor(with_const.values, i) for i in range(with_const.shape[1])],
            index=with_const.columns,
        ).drop("const", errors="ignore")
        worst = vif.idxmax()
        if vif[worst] <= vif_threshold:
            break
        reduced = reduced.drop(columns=[worst])

    return importances, reduced.columns.tolist()


importances, final_features = select_features(X, y)
X_final = X[final_features]

st.header("Feature selection")
col_a, col_b = st.columns(2)
with col_a:
    st.write("Top features by Random Forest importance")
    st.dataframe(importances.head(15).rename("importance"))
with col_b:
    st.write(f"Kept after VIF pruning (VIF ≤ 6): {len(final_features)} features")
    st.write(final_features)


# ----------------------------------------------------------------------
# 5. Baseline model comparison (fast — always runs)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Cross-validating baseline models...")
def baseline_comparison(x: pd.DataFrame, y: pd.Series):
    models = {
        "Linear Regression": LinearRegression(),
        "KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
        "KNN (k=20)": KNeighborsRegressor(n_neighbors=20),
        "Decision Tree (depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42),
        "XGBoost": xgb.XGBRegressor(
            max_depth=2, learning_rate=0.1, n_estimators=100, min_child_weight=5, random_state=42
        ),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, m in models.items():
        scores = -cross_val_score(m, x, y, cv=kf, scoring="neg_mean_absolute_percentage_error", n_jobs=-1)
        rows.append({"Model": name, "Mean MAPE": scores.mean(), "Std MAPE": scores.std()})
    return pd.DataFrame(rows).sort_values("Mean MAPE").reset_index(drop=True)


st.header("Baseline model comparison (5-fold CV)")
st.dataframe(baseline_comparison(X_final, y))

with st.expander("Exhaustive hyperparameter search (slow — can take several minutes)"):
    st.caption("Off by default. The original notebook ran this on every rerun, which is why the app used to hang.")
    if st.button("Run GridSearchCV"):
        param_grids = {
            "KNeighborsRegressor": (KNeighborsRegressor(), {"n_neighbors": [5, 10, 20], "weights": ["uniform", "distance"]}),
            "RandomForestRegressor": (
                RandomForestRegressor(random_state=42),
                {"n_estimators": [50, 100], "max_depth": [5, 10], "min_samples_leaf": [1, 5]},
            ),
            "GradientBoostingRegressor": (
                GradientBoostingRegressor(random_state=42),
                {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            ),
            "XGBoost": (
                xgb.XGBRegressor(random_state=42),
                {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            ),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        results = []
        progress = st.progress(0.0)
        for i, (name, (model, grid)) in enumerate(param_grids.items()):
            with st.spinner(f"Searching {name}..."):
                gs = GridSearchCV(model, grid, cv=kf, scoring="neg_mean_absolute_percentage_error", n_jobs=-1)
                gs.fit(X_final, y)
                results.append({"Model": name, "Best MAPE": -gs.best_score_, "Best Params": gs.best_params_})
            progress.progress((i + 1) / len(param_grids))
        st.dataframe(pd.DataFrame(results))


# ----------------------------------------------------------------------
# 6. Train/test split (balanced target distribution, like the original)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Finding a balanced train/test split...")
def best_split(x: pd.DataFrame, y: pd.Series, test_size=0.2, attempts=200, n_bins=20, random_state=0):
    rng = np.random.RandomState(random_state)
    bins = np.linspace(y.min(), y.max(), n_bins)
    best_mse, best = np.inf, None
    for _ in range(attempts):
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=int(rng.randint(0, 1_000_000)))
        h_tr, _ = np.histogram(y_tr, bins=bins, density=True)
        h_te, _ = np.histogram(y_te, bins=bins, density=True)
        mse = float(np.mean((h_tr - h_te) ** 2))
        if mse < best_mse:
            best_mse, best = mse, (x_tr, x_te, y_tr, y_te)
    return best


X_train, X_test, y_train, y_test = best_split(X_final, y)


# ----------------------------------------------------------------------
# 7. Final classical models
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Training Linear Regression / Random Forest / XGBoost...")
def train_classical_models(x_train: pd.DataFrame, y_train: pd.Series):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(
            max_depth=3, learning_rate=0.1, n_estimators=200, min_child_weight=5, random_state=42
        ),
    }
    for m in models.values():
        m.fit(x_train, y_train)
    return models


trained_models = train_classical_models(X_train, y_train)

st.header("Final model performance")
metric_rows = []
for name, m in trained_models.items():
    y_tr_pred = m.predict(X_train)
    y_te_pred = m.predict(X_test)
    metric_rows.append(
        {
            "Model": name,
            "Train MAE": mean_absolute_error(y_train, y_tr_pred),
            "Test MAE": mean_absolute_error(y_test, y_te_pred),
            "Train R2": r2_score(y_train, y_tr_pred),
            "Test R2": r2_score(y_test, y_te_pred),
        }
    )
st.dataframe(pd.DataFrame(metric_rows))


# ----------------------------------------------------------------------
# 8. Neural network — with sliders for the training params
# ----------------------------------------------------------------------
st.header("Neural network")

if not TF_AVAILABLE:
    st.error(
        "TensorFlow isn't installed, so the neural network section is disabled. "
        f"Add `tensorflow` to requirements.txt and redeploy. ({TF_IMPORT_ERROR})"
    )
else:
    nn_c1, nn_c2, nn_c3, nn_c4 = st.columns(4)
    with nn_c1:
        nn_epochs = st.slider("Epochs", min_value=20, max_value=500, value=150, step=10)
    with nn_c2:
        nn_dropout = st.slider("Dropout rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
    with nn_c3:
        nn_lr = st.select_slider("Learning rate", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)
    with nn_c4:
        nn_batch = st.select_slider("Batch size", options=[16, 32, 64, 128, 256], value=64)

    def train_nn_model(x_train, y_train, x_test, y_test, epochs, dropout_rate, learning_rate, batch_size):
        # Import exclusively from tensorflow.keras (never the standalone `keras`
        # package) so there's only one Keras implementation in play.
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        model = Sequential(
            [
                Input(shape=(x_train.shape[1],)),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(32, activation="relu"),
                Dense(1, activation="relu"),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mae")

        early_stopping = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, y_test),
            verbose=0,
            callbacks=[early_stopping],
        )
        return model, history

    if "nn_model" not in st.session_state:
        st.session_state.nn_model = None
        st.session_state.nn_history = None

    if st.button("Train neural network"):
        with st.spinner("Training..."):
            model, history = train_nn_model(
                X_train.values, y_train.values, X_test.values, y_test.values, nn_epochs, nn_dropout, nn_lr, nn_batch
            )
            st.session_state.nn_model = model
            st.session_state.nn_history = history

    if st.session_state.nn_model is not None:
        nn_model = st.session_state.nn_model
        nn_history = st.session_state.nn_history
        y_tr_pred = nn_model.predict(X_train.values, verbose=0).ravel()
        y_te_pred = nn_model.predict(X_test.values, verbose=0).ravel()

        c1, c2 = st.columns(2)
        c1.metric("Test MAE", f"{mean_absolute_error(y_test, y_te_pred):.3f}")
        c2.metric("Test R²", f"{r2_score(y_test, y_te_pred):.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(nn_history.history["loss"], label="Train loss")
        axes[0].plot(nn_history.history["val_loss"], label="Val loss")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MAE loss")
        axes[0].legend()
        axes[1].scatter(y_train, y_tr_pred, s=6, alpha=0.5, label="Train")
        axes[1].scatter(y_test, y_te_pred, s=6, alpha=0.5, label="Test")
        axes[1].set_xlabel("Actual")
        axes[1].set_ylabel("Predicted")
        axes[1].legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Adjust the sliders above and click **Train neural network** to fit the model.")


# ----------------------------------------------------------------------
# 9. Interactive prediction — sliders for the predictive model
# ----------------------------------------------------------------------
st.header("Try it yourself — predict life expectancy")

model_options = list(trained_models.keys())
if TF_AVAILABLE and st.session_state.get("nn_model") is not None:
    model_options.append("Neural Network")
chosen_model_name = st.selectbox("Model", model_options)

num_features_in_final = [f for f in final_features if f in num_cols]
cat_groups = {}
for c in cat_cols:
    matched = [f for f in final_features if f.startswith(f"{c}_")]
    if matched:
        cat_groups[c] = matched

input_values = {}
slider_cols = st.columns(2)
for i, col in enumerate(num_features_in_final):
    lo, hi = float(raw_numeric[col].min()), float(raw_numeric[col].max())
    default = float(raw_numeric[col].median())
    with slider_cols[i % 2]:
        input_values[col] = st.slider(col, min_value=lo, max_value=hi, value=default, key=f"pred_{col}")

for col, dummy_cols in cat_groups.items():
    labels = [c[len(col) + 1:] for c in dummy_cols] + ["(reference / other)"]
    choice = st.selectbox(col, labels, index=len(labels) - 1, key=f"pred_cat_{col}")
    for dc in dummy_cols:
        input_values[dc] = 0
    if choice != "(reference / other)":
        input_values[dummy_cols[labels.index(choice)]] = 1

if st.button("Predict"):
    full_raw = raw_numeric.median().copy()
    for c in num_features_in_final:
        full_raw[c] = input_values[c]
    scaled_full = scaler.transform(full_raw.to_frame().T)[0]
    scaled_map = dict(zip(num_cols, scaled_full))

    row = {}
    for f in final_features:
        row[f] = scaled_map[f] if f in num_cols else input_values.get(f, 0)
    X_input = pd.DataFrame([row])[final_features]

    if chosen_model_name == "Neural Network":
        pred = float(st.session_state.nn_model.predict(X_input.values, verbose=0)[0][0])
    else:
        pred = float(trained_models[chosen_model_name].predict(X_input)[0])

    st.metric("Predicted life expectancy", f"{pred:.1f} years")


# ----------------------------------------------------------------------
# 10. Test-set predictions by country (if identifier columns exist)
# ----------------------------------------------------------------------
if "Country Name" in data_drop.columns:
    st.header("Test-set predictions by country")
    lookup_model = trained_models["Random Forest"]
    test_preds = lookup_model.predict(X_test)
    results_df = pd.DataFrame(
        {
            "Country Name": data_drop.loc[X_test.index, "Country Name"].values,
            "Year": data_drop.loc[X_test.index, "Year"].values if "Year" in data_drop.columns else None,
            "Actual": y_test.values,
            "Predicted": test_preds,
        }
    )
    results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
    country_choice = st.selectbox("Country", sorted(results_df["Country Name"].unique()))
    st.dataframe(results_df[results_df["Country Name"] == country_choice].sort_values("Year"))


# ----------------------------------------------------------------------
# 11. SHAP explainability
# ----------------------------------------------------------------------
st.header("SHAP explainability")

shap_model_name = st.selectbox("Model to explain", list(trained_models.keys()), key="shap_model")
if st.button("Compute SHAP values"):
    with st.spinner("Computing SHAP values..."):
        model = trained_models[shap_model_name]
        background = X_train.sample(min(100, len(X_train)), random_state=42)
        sample = X_test.sample(min(100, len(X_test)), random_state=42)

        if shap_model_name in ("Random Forest", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample, check_additivity=False)
            expected_value = explainer.expected_value
        else:
            explainer = shap.Explainer(model, background)
            explanation = explainer(sample)
            shap_values = explanation.values
            expected_value = explanation.base_values[0] if hasattr(explanation, "base_values") else explainer.expected_value

        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.ravel(expected_value)[0]

        st.session_state.shap_values = shap_values
        st.session_state.shap_sample = sample
        st.session_state.shap_expected_value = expected_value

if st.session_state.get("shap_values") is not None:
    shap_values = st.session_state.shap_values
    sample = st.session_state.shap_sample
    expected_value = st.session_state.shap_expected_value

    st.subheader("Feature importance (mean |SHAP value|)")
    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader("Summary (beeswarm)")
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader("Single prediction breakdown")
    row_idx = st.slider("Test-set row", 0, len(sample) - 1, 0)
    plt.figure()
    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=expected_value,
        data=sample.iloc[row_idx].values,
        feature_names=sample.columns.tolist(),
    )
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(plt.gcf())
    plt.close()
else:
    st.info("Click **Compute SHAP values** to explain the selected model's predictions.")
ity_prediction_test)
