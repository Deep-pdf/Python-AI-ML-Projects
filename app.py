import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Cart Abandonment Analysis", layout="wide")

st.title("🛒 Online Shopping Cart Abandonment Analysis")

st.markdown(
    """
    Upload an e-commerce CSV file and automatically get:
    - Category-wise abandonment
    - Visualizations
    - Logistic Regression analysis
    - Heatmap insights
    """
)

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Column Selection
    # ---------------------------
    st.subheader("🔧 Select Required Columns")

    col1, col2, col3 = st.columns(3)

    with col1:
        price_col = st.selectbox("Unit Price Column", df.columns)

    with col2:
        qty_col = st.selectbox("Quantity Column", df.columns)

    with col3:
        purchase_col = st.selectbox("Purchased Column (0/1)", df.columns)

    category_col = st.selectbox("Product Category Column", df.columns)

    # ---------------------------
    # Processing
    # ---------------------------
    if st.button("🚀 Run Analysis"):

        data = df[[category_col, price_col, qty_col, purchase_col]].copy()

        data.columns = ["category", "unit_price", "quantity", "purchased"]

        # Cart Value
        data["cart_value"] = data["unit_price"] * data["quantity"]

        # Abandonment
        data["aband"] = 1 - data["purchased"]

        # ---------------------------
        # Category-wise Abandonment
        # ---------------------------
        cat_aband = (
            data.groupby("category")["aband"]
            .mean() * 100
        ).reset_index()

        st.subheader("📊 Category-wise Abandonment Rate")
        st.dataframe(cat_aband)

        fig1, ax1 = plt.subplots(figsize=(8,5))
        sns.barplot(data=cat_aband, x="category", y="aband", ax=ax1)
        ax1.set_title("Category-wise Abandonment Rate (%)")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

        # ---------------------------
        # Count Plot
        # ---------------------------
        fig2, ax2 = plt.subplots(figsize=(5,4))
        sns.countplot(x="aband", data=data, ax=ax2)
        ax2.set_xticks([0,1])
        ax2.set_xticklabels(["Completed", "Abandoned"])
        ax2.set_title("Abandoned vs Completed")
        st.pyplot(fig2)

        # ---------------------------
        # Summary
        # ---------------------------
        st.subheader("📈 Summary Statistics")
        st.write(data["cart_value"].describe())
        st.write(f"Overall Abandonment Rate: {data['aband'].mean()*100:.2f}%")

        # ---------------------------
        # Logistic Regression
        # ---------------------------
        X = data[["cart_value"]]
        y = data["aband"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:,1]

        st.subheader("🤖 Logistic Regression Analysis")
        st.write("R² Score:", r2_score(y_test, y_prob))

        # ---------------------------
        # Probability Curve
        # ---------------------------
        sorted_idx = np.argsort(X_test.values.ravel())
        x_sorted = X_test.values.ravel()[sorted_idx]
        y_sorted = y_prob[sorted_idx]

        fig3, ax3 = plt.subplots(figsize=(8,5))
        ax3.scatter(X_test, y_test, alpha=0.4)
        ax3.plot(x_sorted, y_sorted, color="red")
        ax3.set_xlabel("Cart Value")
        ax3.set_ylabel("Abandonment Probability")
        ax3.set_title("Cart Value vs Abandonment Probability")
        st.pyplot(fig3)

        # ---------------------------
        # Heatmap (One Feature)
        # ---------------------------
        heat_df = pd.DataFrame({
            "cart_value": X_test["cart_value"],
            "abandon_prob": y_prob
        })

        heat_df["cart_bin"] = pd.cut(
            heat_df["cart_value"],
            bins=6,
            labels=["Very Low","Low","Medium",
                    "Medium-High","High","Very High"]
        )

        heatmap_data = (
            heat_df.groupby("cart_bin")["abandon_prob"]
            .mean().to_frame()
        )

        fig4, ax4 = plt.subplots(figsize=(6,4))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax4
        )
        ax4.set_title("Cart Value vs Abandonment Probability (Heatmap)")
        st.pyplot(fig4)
