from llama_index.core.node_parser import SentenceSplitter


def get_text_splitter():

    splitter = SentenceSplitter(
        chunk_size=1200,
        chunk_overlap=100,
    )

    return splitter