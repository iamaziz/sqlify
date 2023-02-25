import pandas as pd
import streamlit as st
import streamlit_ace as stace
import duckdb
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="SQLify", page_icon="🔎", layout="wide")
st.title("SQLify")


@st.cache_data
def _read_csv(f, **kwargs):
    df = pd.read_csv(f, on_bad_lines="skip", **kwargs)
    # clean
    df.columns = [c.strip() for c in df.columns]
    return df


SAMPLE_DATA = {
    "Churn dataset": "https://raw.githubusercontent.com/AtashfarazNavid/MachineLearing-ChurnModeling/main/Streamlit-WebApp-1/Churn.csv",
    "Periodic Table": "https://gist.githubusercontent.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee/raw/1d92663004489a5b6926e944c1b3d9ec5c40900e/Periodic%2520Table%2520of%2520Elements.csv",
    "Movies": "https://raw.githubusercontent.com/reisanar/datasets/master/HollywoodMovies.csv",
    "Iris Flower": "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv",
}


def read_data():
    txt = "Upload a data file (supported files: .csv, .tsv, .xls, .xlsx)"
    placeholder = st.empty()
    with placeholder:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            file_ = st.file_uploader(txt)
        with col2:
            url = st.text_input(
                "Read from a URL",
                placeholder="Enter URL (supported types: .csv and .tsv)",
            )
            if url:
                file_ = url
        with col3:
            selected = st.selectbox("", options=[""] + list(SAMPLE_DATA))
            if selected:
                file_ = SAMPLE_DATA[selected]

    if not file_:
        st.stop()

    placeholder.empty()

    try:
        return _read_csv(file_)
    except Exception as e:
        st.warning("Unsupported file type!")
        st.stop()


def display(df):
    view_info = st.checkbox("view data types")
    st.dataframe(df, use_container_width=True)

    # info
    st.markdown(f"> <sup>shape `{df.shape}`</sup>", unsafe_allow_html=True)

    if view_info:
        types_ = df.dtypes.to_dict()
        types_ = [{"Column": c, "Type": t} for c, t in types_.items()]
        df_ = pd.DataFrame(types_)
        st.sidebar.subheader("TABLE DETAILS")
        st.sidebar.write(df_)


def code_editor():
    # Spawn a new Ace editor
    col1, col2 = st.columns([4, 1])
    with col2:
        # configs
        _THEMES = stace.THEMES
        _KEYBINDINGS = stace.KEYBINDINGS
        theme = st.selectbox("Theme", options=[_THEMES[2]] + _THEMES)
        keybinding = st.selectbox(
            "Keybinding", options=[_KEYBINDINGS[-2]] + _KEYBINDINGS
        )
        font_size = st.slider("Font size", min_value=5, max_value=24, value=14)
        tab_size = st.slider("Tabe size", min_value=1, max_value=8, value=4)
        # kwargs = {theme: theme, keybinding: keybinding} # TODO: DRY
    with col1:
        hint = "Type SQL to query the table above. By default, the table name is 'df'. For example, to select 10 rows: \nSELECT * FROM df LIMIT 10"
        content = stace.st_ace(
            language="sql",
            height=350,
            annotations="ANNOOO ",
            placeholder=hint,
            keybinding=keybinding,
            theme=theme,
            font_size=font_size,
            tab_size=tab_size,
        )

    # Display editor's content as you type
    # content
    return content


@st.cache_data
def query_data(sql, df):
    try:
        return duckdb.query(sql).df()
    except Exception as e:
        st.warning("Invalid Query!")
        # st.stop()


def download(df, save_as="results.csv"):
    # -- to download
    @st.cache_data
    def convert_df(_df):
        return _df.to_csv().encode("utf-8")

    csv = convert_df(df)
    st.download_button(
        "Download",
        csv,
        save_as,
        "text/csv",
    )


def display_results(query, result: pd.DataFrame):
    # st.write("> QUERY")
    # st.code(query, language="sql")
    st.write("> RESULTS")
    st.dataframe(result, use_container_width=True)
    st.markdown(f"> `{result.shape}`")
    download(result)


@st.cache_resource
def data_profiler(df):
    return ProfileReport(df, title="Profiling Report")


def docs():
    content = """
    
    #### Query tabular files (csv, xlsx .. etc) using SQL
    To get started, drag and drop the file or read from a URL. 
    > <sub>[src code here](https://github.com/iamaziz/sqlify)</sub>
    """

    with st.expander("READE"):
        st.markdown(content, unsafe_allow_html=True)


if __name__ == "__main__":
    docs()
    df = read_data()
    display(df)
    st.write("---")
    sql = code_editor()
    if sql:
        res = query_data(sql, df)
        display_results(sql, res)

    if st.checkbox("DATA PROFILE"):
        profile = data_profiler(df)

        st_profile_report(profile)
