import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# APLIKASI SISTEM REKOMENDASI BUKU - BOOK-CROSSING
# Metode: User-Based dan Item-Based Collaborative Filtering
# =========================================================

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System")
st.write(
    "Aplikasi rekomendasi buku menggunakan dataset **Book-Crossing** "
    "dengan metode User-Based dan Item-Based Collaborative Filtering."
)

st.sidebar.header("⚙️ Pengaturan Data")
st.sidebar.write(
    "Unggah tiga file dataset Book-Crossing. "
    "Gunakan file CSV asli dengan pemisah titik koma (;)."
)

users_file = st.sidebar.file_uploader("Upload BX-Users.csv", type=["csv"])
books_file = st.sidebar.file_uploader("Upload BX-Books.csv", type=["csv"])
ratings_file = st.sidebar.file_uploader("Upload BX-Book-Ratings.csv", type=["csv"])

st.sidebar.divider()
min_user_ratings = st.sidebar.number_input(
    "Minimal rating per pengguna", min_value=1, max_value=100, value=5
)
min_book_ratings = st.sidebar.number_input(
    "Minimal rating per buku", min_value=1, max_value=100, value=5
)
max_users = st.sidebar.number_input(
    "Maksimal pengguna yang diproses", min_value=100, max_value=5000, value=1200, step=100
)
max_books = st.sidebar.number_input(
    "Maksimal buku yang diproses", min_value=100, max_value=5000, value=1500, step=100
)
n_recommendations = st.sidebar.slider(
    "Jumlah rekomendasi", min_value=5, max_value=20, value=10
)

@st.cache_data(show_spinner=False)
def load_and_prepare(users_bytes, books_bytes, ratings_bytes):
    users = pd.read_csv(
        pd.io.common.BytesIO(users_bytes), sep=";", encoding="latin-1"
    )
    books = pd.read_csv(
        pd.io.common.BytesIO(books_bytes), sep=";", encoding="latin-1", escapechar="\\"
    )
    ratings = pd.read_csv(
        pd.io.common.BytesIO(ratings_bytes), sep=";", encoding="latin-1"
    )

    # Pembersihan data mengikuti notebook
    users = users.drop_duplicates().copy()
    books = books.drop_duplicates(subset=["ISBN"]).copy()
    ratings = ratings.drop_duplicates(
        subset=["User-ID", "ISBN"], keep="last"
    ).copy()

    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"], errors="coerce"
    )
    ratings = ratings.dropna(
        subset=["User-ID", "ISBN", "Book-Rating"]
    ).copy()
    ratings["ISBN"] = ratings["ISBN"].astype(str)

    # Rating 0 tidak digunakan; hanya rating eksplisit 1-10
    explicit_ratings = ratings[ratings["Book-Rating"] > 0].copy()

    ratings_books = explicit_ratings.merge(
        books[["ISBN", "Book-Title", "Book-Author"]],
        on="ISBN",
        how="left"
    )

    return users, books, ratings, explicit_ratings, ratings_books


@st.cache_data(show_spinner=False)
def build_user_item_matrix(explicit_ratings, min_user, min_book, max_user, max_book):
    user_counts = explicit_ratings["User-ID"].value_counts()
    selected_users = user_counts[user_counts >= min_user].index
    cf_data = explicit_ratings[
        explicit_ratings["User-ID"].isin(selected_users)
    ].copy()

    book_counts = cf_data["ISBN"].value_counts()
    selected_books = book_counts[book_counts >= min_book].index
    cf_data = cf_data[cf_data["ISBN"].isin(selected_books)].copy()

    # Batasi pengguna dan buku agar komputasi tetap ringan
    top_user_ids = cf_data["User-ID"].value_counts().head(max_user).index
    data_sample = cf_data[cf_data["User-ID"].isin(top_user_ids)].copy()

    top_book_isbns = data_sample["ISBN"].value_counts().head(max_book).index
    data_sample = data_sample[data_sample["ISBN"].isin(top_book_isbns)].copy()

    matrix = data_sample.pivot_table(
        index="User-ID",
        columns="ISBN",
        values="Book-Rating",
        aggfunc="mean"
    ).fillna(0).astype("float32")

    return cf_data, data_sample, matrix


def get_user_recommendations(user_id, matrix, similarity_df, books, n):
    if user_id not in matrix.index:
        return pd.DataFrame()

    similarities = similarity_df.loc[user_id].drop(user_id)
    target_ratings = matrix.loc[user_id]
    unrated_isbns = target_ratings[target_ratings == 0].index

    scores = {}
    for isbn in unrated_isbns:
        other_ratings = matrix[isbn]
        rated_by_others = other_ratings[other_ratings > 0].index
        common_users = similarities.index.intersection(rated_by_others)

        if len(common_users) == 0:
            continue

        sim = similarities.loc[common_users]
        ratings_for_book = other_ratings.loc[common_users]
        denominator = sim.abs().sum()

        if denominator > 0:
            scores[isbn] = (sim * ratings_for_book).sum() / denominator

    if not scores:
        return pd.DataFrame()

    result = (
        pd.Series(scores, name="Predicted Rating")
        .sort_values(ascending=False)
        .head(n)
        .rename_axis("ISBN")
        .reset_index()
    )
    return result.merge(
        books[["ISBN", "Book-Title", "Book-Author"]],
        on="ISBN",
        how="left"
    )


def get_item_recommendations(user_id, matrix, item_similarity_df, books, n):
    if user_id not in matrix.index:
        return pd.DataFrame()

    user_ratings = matrix.loc[user_id]
    rated_books = user_ratings[user_ratings > 0]

    if rated_books.empty:
        return pd.DataFrame()

    unrated_isbns = user_ratings[user_ratings == 0].index
    scores = {}

    for target_isbn in unrated_isbns:
        similarities = item_similarity_df.loc[target_isbn, rated_books.index]
        denominator = similarities.abs().sum()

        if denominator > 0:
            scores[target_isbn] = (
                similarities * rated_books
            ).sum() / denominator

    if not scores:
        return pd.DataFrame()

    result = (
        pd.Series(scores, name="Predicted Rating")
        .sort_values(ascending=False)
        .head(n)
        .rename_axis("ISBN")
        .reset_index()
    )
    return result.merge(
        books[["ISBN", "Book-Title", "Book-Author"]],
        on="ISBN",
        how="left"
    )


if not (users_file and books_file and ratings_file):
    st.info("👈 Silakan unggah ketiga dataset pada sidebar untuk memulai aplikasi.")
    st.markdown(
        """
        **File yang dibutuhkan:**
        - `BX-Users.csv`
        - `BX-Books.csv`
        - `BX-Book-Ratings.csv`

        Aplikasi menggunakan rating eksplisit (nilai 1–10). Rating 0
        diperlakukan sebagai interaksi tanpa penilaian angka dan tidak
        dimasukkan ke proses rekomendasi.
        """
    )
    st.stop()

try:
    with st.spinner("Membaca dan membersihkan dataset..."):
        users, books, ratings, explicit_ratings, ratings_books = load_and_prepare(
            users_file.getvalue(),
            books_file.getvalue(),
            ratings_file.getvalue()
        )

    with st.spinner("Menyusun matriks pengguna dan buku..."):
        cf_data, data_sample, user_item_matrix = build_user_item_matrix(
            explicit_ratings,
            int(min_user_ratings),
            int(min_book_ratings),
            int(max_users),
            int(max_books)
        )

    if user_item_matrix.empty or user_item_matrix.shape[0] < 2 or user_item_matrix.shape[1] < 2:
        st.error(
            "Data hasil penyaringan terlalu sedikit untuk membuat rekomendasi. "
            "Coba turunkan minimal rating pengguna/buku atau naikkan batas jumlah data."
        )
        st.stop()

    # Informasi dataset
    st.subheader("1. Ringkasan Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Users", f"{len(users):,}")
    c2.metric("Jumlah Books", f"{len(books):,}")
    c3.metric("Seluruh Ratings", f"{len(ratings):,}")
    c4.metric("Explicit Ratings", f"{len(explicit_ratings):,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Pengguna pada Matriks", f"{user_item_matrix.shape[0]:,}")
    c6.metric("Buku pada Matriks", f"{user_item_matrix.shape[1]:,}")
    c7.metric("Interaksi Terpilih", f"{len(data_sample):,}")

    with st.expander("Lihat contoh data buku dan rating"):
        st.dataframe(
            ratings_books[["User-ID", "ISBN", "Book-Title", "Book-Author", "Book-Rating"]].head(10),
            use_container_width=True
        )

    st.subheader("2. Distribusi Rating Eksplisit")
    rating_distribution = (
        explicit_ratings["Book-Rating"]
        .value_counts()
        .sort_index()
        .rename_axis("Rating")
        .reset_index(name="Jumlah")
    )
    st.bar_chart(rating_distribution.set_index("Rating"))

    st.subheader("3. Pilih Pengguna")
    available_users = user_item_matrix.index.tolist()
    default_user = int(
        (user_item_matrix > 0).sum(axis=1).sort_values(ascending=False).index[0]
    )
    user_id = st.selectbox(
        "Pilih User-ID untuk mendapatkan rekomendasi",
        options=available_users,
        index=available_users.index(default_user),
        format_func=lambda x: str(x)
    )

    rated_count = int((user_item_matrix.loc[user_id] > 0).sum())
    st.caption(f"Pengguna {user_id} memiliki {rated_count} rating pada data yang diproses.")

    # Hitung similarity hanya setelah pengguna dipilih dan data tersedia
    with st.spinner("Menghitung kemiripan pengguna dan buku..."):
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )

        item_similarity = cosine_similarity(user_item_matrix.T)
        item_similarity_df = pd.DataFrame(
            item_similarity,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )

    st.subheader("4. Rekomendasi Buku")
    tab_user, tab_item = st.tabs([
        "👥 User-Based Collaborative Filtering",
        "📖 Item-Based Collaborative Filtering"
    ])

    with tab_user:
        st.write(
            "Rekomendasi dihitung dari kemiripan pola rating pengguna ini "
            "dengan pengguna lain."
        )
        user_result = get_user_recommendations(
            user_id, user_item_matrix, user_similarity_df, books, n_recommendations
        )
        if user_result.empty:
            st.warning("Belum ada rekomendasi User-Based untuk pengguna ini.")
        else:
            user_result["Predicted Rating"] = user_result["Predicted Rating"].round(2)
            st.dataframe(
                user_result[["Book-Title", "Book-Author", "ISBN", "Predicted Rating"]],
                use_container_width=True,
                hide_index=True
            )

    with tab_item:
        st.write(
            "Rekomendasi dihitung dari kemiripan buku yang belum dinilai "
            "dengan buku-buku yang pernah dinilai pengguna."
        )
        item_result = get_item_recommendations(
            user_id, user_item_matrix, item_similarity_df, books, n_recommendations
        )
        if item_result.empty:
            st.warning("Belum ada rekomendasi Item-Based untuk pengguna ini.")
        else:
            item_result["Predicted Rating"] = item_result["Predicted Rating"].round(2)
            st.dataframe(
                item_result[["Book-Title", "Book-Author", "ISBN", "Predicted Rating"]],
                use_container_width=True,
                hide_index=True
            )

    st.subheader("5. Buku dengan Rating Terbanyak")
    popular_books = (
        ratings_books.groupby(["ISBN", "Book-Title", "Book-Author"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="Jumlah Rating")
    )
    st.dataframe(popular_books, use_container_width=True, hide_index=True)

    st.caption(
        "Catatan: Predicted Rating merupakan skor hasil perhitungan kemiripan "
        "Collaborative Filtering, bukan jaminan rating yang akan diberikan pengguna."
    )

except Exception as e:
    st.error("Terjadi kendala saat memproses data.")
    st.exception(e)
    st.info(
        "Pastikan ketiga file merupakan dataset Book-Crossing asli, "
        "menggunakan pemisah titik koma (;), dan memiliki kolom yang sesuai."
    )
