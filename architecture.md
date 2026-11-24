# 入库流（幂等）
10 个 txt
 → TextParser.load            List[Document]
 → TextSplitter.split          List[Document]（chunk_size=400, overlap=60，写 chunk_index）
 → BgeEmbedding                List[List[float]]（1024 维）
 → build_milvus_data           List[dict]（id = md5(source|chunk_index|content)）
 → MilvusStore.upsert          —— 同一内容重复入库按主键覆盖，不会重复
 ↘ BM25Retriever.build_index   内存倒排

# 查询流
query
 ├─ QueryRewriter（可关）      → [原问题, 子查询1, 子查询2]
 ├─ 每个子查询：embed → Milvus.search(15) + BM25.search(15)  ⇒ 3×2 = 6 个列表
 ├─ rrf_fuse_many(6 个列表)    → 融合 15 条（1/(60+rank) 逐列表累加）
 ├─ BgeReranker.rerank(原问题) → top 5（异常 → 降级用 RRF 前 5）
 ├─ build_prompt                → 来源标注 + 防幻觉指令 + 最大字数上限
 └─ LLMGenerator                → answer

# 四个可替换点（依赖注入的价值）
BaseParser / BaseVectorStore / BaseGenerator / BaseReranker —— 换实现不动 Pipeline
