import sys

from ragkb.config.settings import (
    MILVUS_URI, DB_NAME, COLLECTION_NAME, EMBED_DIM,
    ENCODING, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TEMPERATURE,
    TOP_K, REFUSE_THRESHOLD,
)
from ragkb.config.corpus import corpus_files


from ragkb.embeddings.bge_embedding import  BgeEmbedding
from ragkb.generators.llm_generator import LLMGenerator
from ragkb.parsers.text_parser import TextParser
from ragkb.pipeline.rag_pipeline import RAGPipeline
from ragkb.splitters.text_splitter import TextSplitter
from ragkb.vectorstores.milvus_store import MilvusStore
from ragkb.retrievers.bm25_retriever import BM25Retriever
from ragkb.retrievers.reranker import BgeReranker
from ragkb.retrievers.query_rewriter import QueryRewriter


def build_pipeline() -> RAGPipeline:
    parser = TextParser()
    splitter = TextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
    embedding_model = BgeEmbedding(EMBEDDING_MODEL)
    vector_store = MilvusStore(
        uri =MILVUS_URI,
        db_name = DB_NAME,
        collection_name=COLLECTION_NAME,
        dimension=EMBED_DIM)
    bm25_retriever = BM25Retriever()
    reranker = BgeReranker()

    vector_store.connect()
    vector_store.create_collection()

    generator = LLMGenerator(
        model_name=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )

    query_rewriter = QueryRewriter(generator=generator)

    return RAGPipeline(
        parser=parser,
        splitter=splitter,
        vector_store=vector_store,
        generator=generator,
        embedding_model=embedding_model,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        refuse_threshold=REFUSE_THRESHOLD,
    )


def main():
    pipeline = build_pipeline()
    files = corpus_files()

    if "--ingest" in sys.argv:
        # 首次使用 / 换了语料：真正入库（要算 embedding，345 条约 1~3 分钟）
        count = pipeline.ingest(files, ENCODING)
        print(f"入库完成，共 {count} 条（{len(files)} 个语料文件）")
    else:
        # 日常启动：只重建内存里的 BM25 索引（秒级），向量库里的数据不用重算
        chunks = []
        for f in files:
            chunks.extend(pipeline.splitter.split(pipeline.parser.load(f, ENCODING)))
        pipeline.bm25_retriever.build_index(chunks)
        print(f"BM25 索引已重建（{len(chunks)} 条），跳过向量入库。首次使用请加 --ingest")

    print("RAG 助手已启动，输入 q 退出。")
    while True:
        query = input("\n请输入问题:").strip()
        if query in ("q","quit","exit"):
            break

        if not query:
            continue

        try :
            ans = pipeline.query(query,TOP_K)
            print(f"\n回答：{ans}")
        except Exception as e:
            print(f"\n出错了:{e}")

if __name__ == "__main__":
    main()

