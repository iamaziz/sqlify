import pandas as pd
import pandasql as ps
import streamlit as st
import streamlit_ace as stace

st.set_page_config(page_title="SQLify", page_icon="🔎", layout="wide")
st.title("SQLify")


@st.cache(allow_output_mutation=True)  # TODO: fix
def _read_csv(f, **kwargs):
    df = pd.read_csv(f, on_bad_lines="skip", **kwargs)
    # clean
    df.columns = [c.strip() for c in df.columns]
    return df


def read_data():
    txt = "Upload a data file (supported files: .csv, .tsv, .xls, .xlsx)"
    col1, col2 = st.columns(2)
    with col1:
        placeholder = st.empty()
        with placeholder:
            file_ = st.file_uploader(txt)
    with col2:
        placeholder2 = st.empty()
        with placeholder2:
            url = st.text_input(
                "Or read from a URL",
                placeholder="Enter URL (supported types: .csv and .tsv)",
            )
            if url:
                placeholder2.empty()
                placeholder.empty()
                return _read_csv(url)

    if not file_:
        st.stop()

    placeholder.empty()
    placeholder2.empty()

    if str(file_.name).endswith(".csv"):
        return _read_csv(file_)

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


def query_data(sql):
    try:
        return ps.sqldf(sql)
    except:
        st.warning("Invalid Query!")
        st.stop()


def download(df, save_as="results.csv"):
    # -- to download
    @st.cache
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
        res = query_data(sql)
        display_results(sql, res)
